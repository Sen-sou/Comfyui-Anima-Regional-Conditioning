from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F

import comfy.patcher_extension


WRAPPER_KEY = "comfyui_anima_regional_conditioning"
REGION_TYPE = "ANIMA_CONDITIONING_REGIONS"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnimaConditioningRegionChain:
    previous: Optional["AnimaConditioningRegionChain"]
    mask: torch.Tensor
    conditioning: list
    weight: float

    def flatten(self) -> list["AnimaConditioningRegionChain"]:
        regions = []
        current: Optional[AnimaConditioningRegionChain] = self
        while current is not None:
            regions.append(current)
            current = current.previous
        regions.reverse()
        return regions


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _prepare_mask(mask: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(mask):
        raise RuntimeError(f"Expected mask tensor, got {type(mask)}.")
    mask = mask.detach().float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask[:1]
    elif mask.ndim == 4:
        mask = mask[:1, 0]
    else:
        raise RuntimeError(f"Unsupported mask rank {mask.ndim}; expected H,W or B,H,W.")
    return mask.clamp(0.0, 1.0).cpu().contiguous()


def _extract_conditioning_parts(conditioning: list, name: str) -> tuple[torch.Tensor, dict]:
    if not conditioning:
        raise RuntimeError(f"{name} is empty.")
    first = conditioning[0]
    if not isinstance(first, (list, tuple)) or len(first) < 1:
        raise RuntimeError(f"{name} is not a valid ComfyUI CONDITIONING value.")
    cond = first[0]
    metadata = first[1] if len(first) > 1 and isinstance(first[1], dict) else {}
    if not torch.is_tensor(cond):
        raise RuntimeError(f"{name}[0][0] must be a tensor, got {type(cond)}.")
    cond = cond.detach()
    if cond.ndim == 4 and cond.shape[1] == 1:
        cond = cond.squeeze(1)
    if cond.ndim != 3:
        raise RuntimeError(
            f"{name} cross-attention tensor must have shape B,T,D or B,1,T,D; got {tuple(cond.shape)}."
        )
    return cond, metadata


def _as_batched_ids(ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    ids = ids.to(device=device)
    if ids.ndim == 1:
        return ids.unsqueeze(0)
    if ids.ndim == 2:
        return ids
    raise RuntimeError(f"t5xxl_ids must have rank 1 or 2, got {ids.ndim}.")


def _as_batched_weights(weights: Optional[torch.Tensor], like: torch.Tensor) -> Optional[torch.Tensor]:
    if weights is None:
        return None
    weights = weights.to(device=like.device, dtype=like.dtype)
    if weights.ndim == 1:
        return weights.unsqueeze(0).unsqueeze(-1)
    if weights.ndim == 2:
        return weights.unsqueeze(-1)
    if weights.ndim == 3:
        return weights
    raise RuntimeError(f"t5xxl_weights must have rank 1, 2, or 3, got {weights.ndim}.")


def _normalize_context(context: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Squeeze 4D context [B, 1, S, D] → 3D [B, S, D]. Returns (tensor, was_4d)."""
    if context.ndim == 4 and context.shape[1] == 1:
        return context.squeeze(1), True
    if context.ndim == 3:
        return context, False
    raise RuntimeError(f"Unsupported context shape {tuple(context.shape)}.")


def _match_context_length(context: torch.Tensor, target_len: int) -> torch.Tensor:
    if context.shape[1] == target_len:
        return context
    if context.shape[1] > target_len:
        return context[:, :target_len, :]
    pad = torch.zeros(
        context.shape[0],
        target_len - context.shape[1],
        context.shape[2],
        device=context.device,
        dtype=context.dtype,
    )
    return torch.cat([context, pad], dim=1)


# ---------------------------------------------------------------------------
# Spatial mask -> token masks
# ---------------------------------------------------------------------------

def _masks_to_token_masks(
    masks: list[torch.Tensor],  # each [1, H_mask, W_mask]
    latent_h: int,
    latent_w: int,
    patch_spatial: int,
    temporal_tokens: int,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """
    Resize masks to the DiT token grid and return Flux-style boolean region
    membership [N, S_latent]. Overlaps are preserved; a latent token can belong
    to more than one region.
    """
    padded_h = math.ceil(latent_h / patch_spatial) * patch_spatial
    padded_w = math.ceil(latent_w / patch_spatial) * patch_spatial
    h_tokens = padded_h // patch_spatial
    w_tokens = padded_w // patch_spatial
    spatial_tokens = h_tokens * w_tokens

    resized: list[torch.Tensor] = []
    for mask in masks:
        m = F.interpolate(
            mask.unsqueeze(1),
            size=(h_tokens, w_tokens),
            mode="nearest-exact",
        ).squeeze(1).squeeze(0)
        m = m.reshape(spatial_tokens)
        m = m.unsqueeze(0).expand(temporal_tokens, -1).reshape(-1)
        resized.append(m)

    stacked = torch.stack(resized, dim=0)   # [N, S_latent]
    return stacked > float(threshold)


def _slot_strengths_to_token_strengths(
    masks: torch.Tensor,
    slot_strengths: torch.Tensor,
    default_strength: float,
) -> torch.Tensor:
    strengths = torch.zeros(masks.shape[1], device=masks.device, dtype=slot_strengths.dtype)
    for slot_idx in range(masks.shape[0]):
        strengths = torch.maximum(
            strengths,
            masks[slot_idx].to(slot_strengths.dtype) * slot_strengths[slot_idx],
        )
    return torch.maximum(strengths, torch.full_like(strengths, float(default_strength)))


# ---------------------------------------------------------------------------
# Block-diagonal attention bias builder
# ---------------------------------------------------------------------------

def _build_flux_cross_attention_bias(
    masks: torch.Tensor,        # [N, S_latent] bool
    text_lengths: list[int],    # N entries: [S_base, S_r1, S_r2, ...]
    base_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    mask_strength: float = 1.0,
    slot_strengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    N, S_latent = masks.shape
    S_total = sum(text_lengths)

    if mask_strength <= 0.0:
        return torch.zeros((1, 1, S_latent, S_total), device=device, dtype=dtype)

    masks = masks.to(device=device)
    allowed = torch.zeros((S_latent, S_total), device=device, dtype=torch.bool)

    if slot_strengths is None:
        slot_strengths = torch.ones(N, device=device, dtype=dtype)
    else:
        slot_strengths = slot_strengths.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    offsets = [0]
    for l in text_lengths[:-1]:
        offsets.append(offsets[-1] + l)

    for slot_idx in range(N):
        start = offsets[slot_idx]
        end = start + text_lengths[slot_idx]
        if start == end:
            continue

        if slot_idx == 0 and base_mode == "global":
            allowed[:, start:end] = True
        elif slot_idx == 0 and base_mode == "disabled":
            continue
        else:
            positions = masks[slot_idx].nonzero(as_tuple=True)[0]
            if positions.numel() > 0:
                allowed[positions, start:end] = True

    # SDPA cannot handle rows where every key is -inf. If a token is outside
    # every mask, let it attend to the base text as a numeric and composition
    # fallback. This mirrors the role of background regions in the Flux demo.
    fully_blocked = ~allowed.any(dim=-1)
    if fully_blocked.any() and text_lengths[0] > 0:
        allowed[fully_blocked, :text_lengths[0]] = True

    token_strengths = _slot_strengths_to_token_strengths(
        masks,
        slot_strengths * float(mask_strength),
        default_strength=0.0,
    )
    row_penalties = -12.0 * token_strengths
    bias_2d = torch.where(
        allowed,
        torch.zeros((S_latent, S_total), device=device, dtype=dtype),
        row_penalties[:, None].expand(-1, S_total),
    )
    hard_rows = token_strengths >= 1.0
    if hard_rows.any():
        bias_2d[hard_rows[:, None].expand(-1, S_total) & ~allowed] = float("-inf")

    return bias_2d.unsqueeze(0).unsqueeze(0)


def _build_flux_self_attention_bias(
    masks: torch.Tensor, # [N, S_latent] bool
    base_mode: str,
    mask_strength: float,
    device: torch.device,
    dtype: torch.dtype,
    slot_strengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    N, S_latent = masks.shape
    if mask_strength <= 0.0:
        return torch.zeros((1, 1, S_latent, S_latent), device=device, dtype=dtype)

    m = masks.to(device=device)
    if slot_strengths is None:
        slot_strengths = torch.ones(N, device=device, dtype=dtype)
    else:
        slot_strengths = slot_strengths.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    allowed = torch.zeros((S_latent, S_latent), device=device, dtype=torch.bool)

    for slot_idx in range(N):
        if slot_idx == 0 and base_mode == "disabled":
            continue
        slot = m[slot_idx]
        allowed |= slot[:, None] & slot[None, :]

    if base_mode == "global":
        allowed[:] = True

    # Flux lets uncovered background tokens attend to one another.
    union = torch.zeros(S_latent, device=device, dtype=torch.bool)
    for slot_idx in range(1, N):
        union |= m[slot_idx]
    if base_mode != "disabled":
        union |= m[0]
    background = ~union
    allowed |= background[:, None] & background[None, :]

    allowed |= torch.eye(S_latent, device=device, dtype=torch.bool)

    token_strengths = _slot_strengths_to_token_strengths(
        m,
        slot_strengths * float(mask_strength),
        default_strength=0.0,
    )
    row_penalties = -12.0 * token_strengths
    bias = torch.where(
        allowed,
        torch.zeros((S_latent, S_latent), device=device, dtype=dtype),
        row_penalties[:, None].expand(-1, S_latent),
    )
    hard_rows = token_strengths >= 1.0
    if hard_rows.any():
        bias[hard_rows[:, None].expand(-1, S_latent) & ~allowed] = float("-inf")
    return bias.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Score-level masked attention op (replaces torch_attention_op)
# ---------------------------------------------------------------------------

def _masked_attn_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    transformer_options: Optional[dict] = None,
    attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Drop-in replacement for torch_attention_op that injects a spatial attention
    bias directly into the QK scores before softmax.

    Inputs  (same convention as torch_attention_op):
      q : [B, S_q, n_heads, head_dim]
      k : [B, S_k, n_heads, head_dim]
      v : [B, S_k, n_heads, head_dim]
    attn_bias : [B, 1, S_q, S_k]  (broadcasts over heads)

    Output: [B, S_q, n_heads * head_dim]
    """
    B, Sq, H, D = q.shape

    # [B, S, H, D] → [B, H, S, D] for SDPA
    q_b = q.permute(0, 2, 1, 3)
    k_b = k.permute(0, 2, 1, 3)
    v_b = v.permute(0, 2, 1, 3)

    bias: Optional[torch.Tensor] = None
    if attn_bias is not None:
        # Move to compute device/dtype; [B, 1, S_q, S_k] broadcasts over H.
        bias = attn_bias.to(device=q.device, dtype=q.dtype)

    # PyTorch SDPA: if bias has -inf entries, those positions get ~0 weight after
    # softmax — hard spatial boundaries rather than a soft output blend.
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=bias)

    # [B, H, Sq, D] → [B, Sq, H*D]
    return out.permute(0, 2, 1, 3).reshape(B, Sq, H * D)


# ---------------------------------------------------------------------------
# Patch object
# ---------------------------------------------------------------------------

class AnimaRegionalConditioningPatch:
    def __init__(
        self,
        regions: list[AnimaConditioningRegionChain],
        base_mode: str,
        base_strength: float,
        start_sigma: float,
        end_sigma: float,
        cross_mask_strength: float,
        self_mask_strength: float,
        base_ratio: float,
        cross_inject_every_n_blocks: int,
        self_inject_every_n_blocks: int,
        background_conditioning: Optional[list] = None,
    ):
        if not regions:
            raise RuntimeError("At least one Anima conditioning region is required.")

        self.base_mode = base_mode
        self.base_strength = max(float(base_strength), 0.0)
        self.start_sigma = float(start_sigma)
        self.end_sigma = float(end_sigma)
        self.cross_mask_strength = max(0.0, min(float(cross_mask_strength), 1.0))
        self.self_mask_strength = max(0.0, min(float(self_mask_strength), 1.0))
        self.base_ratio = max(0.0, min(float(base_ratio), 1.0))
        self.cross_inject_every_n_blocks = max(1, int(cross_inject_every_n_blocks))
        self.self_inject_every_n_blocks = max(1, int(self_inject_every_n_blocks))

        # Raw per-region masks [1, H, W] and conditionings.
        # Any positive mask value is considered region membership
        # after token-grid resize, which matches Flux-style boolean routing.
        self.region_masks: list[torch.Tensor] = []
        self.region_weights: list[float] = []
        self.region_conditionings: list[tuple[torch.Tensor, dict]] = []
        self.background_conditioning: Optional[tuple[torch.Tensor, dict]] = None

        if background_conditioning is not None:
            cond, metadata = _extract_conditioning_parts(
                background_conditioning, "background_conditioning"
            )
            self.background_conditioning = (cond.detach().float().cpu().contiguous(), metadata.copy())

        for idx, region in enumerate(regions, start=1):
            weight = max(float(region.weight), 0.0)
            mask = _prepare_mask(region.mask) if weight > 0.0 else torch.zeros_like(_prepare_mask(region.mask))
            cond, metadata = _extract_conditioning_parts(
                region.conditioning, f"region_{idx}.conditioning"
            )
            self.region_masks.append(mask)
            self.region_weights.append(weight)
            self.region_conditionings.append(
                (cond.detach().float().cpu().contiguous(), metadata.copy())
            )

    def prepare_region_conds(
        self,
        diffusion_model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Preprocess region conditionings (T5XXL aware)."""
        prepared: list[torch.Tensor] = []
        for cond, metadata in self.region_conditionings:
            cond = cond.to(device=device, dtype=dtype)
            t5xxl_ids = metadata.get("t5xxl_ids", None)
            if t5xxl_ids is not None and hasattr(diffusion_model, "preprocess_text_embeds"):
                if not torch.is_tensor(t5xxl_ids):
                    raise RuntimeError(f"t5xxl_ids must be a tensor, got {type(t5xxl_ids)}.")
                t5xxl_weights = metadata.get("t5xxl_weights", None)
                if t5xxl_weights is not None and not torch.is_tensor(t5xxl_weights):
                    raise RuntimeError(f"t5xxl_weights must be a tensor, got {type(t5xxl_weights)}.")
                cond = diffusion_model.preprocess_text_embeds(
                    cond,
                    _as_batched_ids(t5xxl_ids, device),
                    t5xxl_weights=_as_batched_weights(t5xxl_weights, cond),
                )
            prepared.append(cond)
        return prepared

    def prepare_background_cond(
        self,
        diffusion_model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.background_conditioning is None:
            return None
        cond, metadata = self.background_conditioning
        cond = cond.to(device=device, dtype=dtype)
        t5xxl_ids = metadata.get("t5xxl_ids", None)
        if t5xxl_ids is not None and hasattr(diffusion_model, "preprocess_text_embeds"):
            if not torch.is_tensor(t5xxl_ids):
                raise RuntimeError(f"t5xxl_ids must be a tensor, got {type(t5xxl_ids)}.")
            t5xxl_weights = metadata.get("t5xxl_weights", None)
            if t5xxl_weights is not None and not torch.is_tensor(t5xxl_weights):
                raise RuntimeError(f"t5xxl_weights must be a tensor, got {type(t5xxl_weights)}.")
            cond = diffusion_model.preprocess_text_embeds(
                cond,
                _as_batched_ids(t5xxl_ids, device),
                t5xxl_weights=_as_batched_weights(t5xxl_weights, cond),
            )
        return cond

    def is_active(self, transformer_options: dict) -> bool:
        sigmas = transformer_options.get("sigmas", None)
        if sigmas is None or not torch.is_tensor(sigmas) or sigmas.numel() == 0:
            return True
        sigma = float(sigmas.max().detach().cpu().item())
        low = min(self.start_sigma, self.end_sigma)
        high = max(self.start_sigma, self.end_sigma)
        return low <= sigma <= high


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

def _validate_anima_model(model) -> torch.nn.Module:
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise RuntimeError("Invalid MODEL input: missing internal .model object.")
    model_config = getattr(base_model, "model_config", None)
    if model_config is not None:
        image_model = model_config.unet_config.get("image_model", None)
        if image_model != "anima":
            raise RuntimeError("Anima Regional Conditioning supports only Anima models.")
    dit = getattr(base_model, "diffusion_model", None)
    if dit is None or not hasattr(dit, "blocks"):
        raise RuntimeError("MODEL does not look like an Anima/Cosmos Predict2 MiniTrainDIT model.")
    if not hasattr(dit, "patch_spatial"):
        raise RuntimeError("Anima diffusion model is missing patch_spatial.")
    return dit


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

def _diffusion_model_wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options", None)
    if not isinstance(transformer_options, dict):
        return executor(*args, **kwargs)

    patch: Optional[AnimaRegionalConditioningPatch] = transformer_options.get(WRAPPER_KEY, None)
    if patch is None:
        return executor(*args, **kwargs)
    if not patch.is_active(transformer_options):
        return executor(*args, **kwargs)
    if patch.base_ratio >= 1.0 or (patch.cross_mask_strength <= 0.0 and patch.self_mask_strength <= 0.0):
        return executor(*args, **kwargs)

    diffusion_model = executor.class_obj

    # ---- Latent geometry ------------------------------------------------
    input_x = args[0] if args else kwargs.get("x", None)
    if input_x is None or input_x.ndim < 5:
        raise RuntimeError("Anima Regional Conditioning expected latent input shaped B,C,T,H,W.")

    latent_h = int(input_x.shape[-2])
    latent_w = int(input_x.shape[-1])
    latent_t = int(input_x.shape[2])
    patch_spatial = int(getattr(diffusion_model, "patch_spatial", 2))
    patch_temporal = int(getattr(diffusion_model, "patch_temporal", 1))

    # ---- Context --------------------------------------------------------
    # Always normalize to 3D [B, S, D].  Passing 3D context to Attention means
    # k_proj produces [B, S, inner_dim] → rearrange → [B, S, n_heads, head_dim],
    # which our _masked_attn_op expects.  (torch_attention_op handles both 4D
    # and 5D via einops "..." patterns, but our replacement handles 4D only.)
    raw_context = args[2] if len(args) > 2 else kwargs.get("context", None)
    if raw_context is None or not torch.is_tensor(raw_context):
        raise RuntimeError("Anima Regional Conditioning expected a tensor context input.")
    context, _ = _normalize_context(raw_context)

    device = context.device
    dtype = context.dtype
    B_total = context.shape[0]
    S_base = context.shape[1]

    # ---- CFG split ------------------------------------------------------
    cond_or_unconds = transformer_options.get("cond_or_uncond", [])
    if not cond_or_unconds:
        return executor(*args, **kwargs)

    num_chunks = len(cond_or_unconds)
    if B_total % num_chunks != 0:
        return executor(*args, **kwargs)
    batch_size = B_total // num_chunks

    background_cond = patch.prepare_background_cond(diffusion_model, device, dtype)
    background_cond_batched: Optional[torch.Tensor] = None
    if background_cond is not None:
        if background_cond.shape[0] == 1:
            background_cond_batched = background_cond.expand(batch_size, -1, -1)
        elif background_cond.shape[0] == batch_size:
            background_cond_batched = background_cond
        else:
            raise RuntimeError(
                f"Background conditioning batch {background_cond.shape[0]} does not match sampler batch {batch_size}."
            )

    # ---- Prepare region conditionings ----------------------------------
    region_conds = patch.prepare_region_conds(diffusion_model, device, dtype)
    region_lengths = [rc.shape[1] for rc in region_conds]

    # Expand to batch_size (region conds usually have B=1)
    region_conds_batched: list[torch.Tensor] = []
    for rc in region_conds:
        if rc.shape[0] == 1:
            rc = rc.expand(batch_size, -1, -1)
        elif rc.shape[0] != batch_size:
            raise RuntimeError(
                f"Region conditioning batch {rc.shape[0]} does not match sampler batch {batch_size}."
            )
        region_conds_batched.append(rc)

    # ---- Build unified context -----------------------------------------
    # Text layout in the unified sequence (per cond chunk):
    #   [base_context | region_1 | region_2 | ... | region_N]
    #
    # For uncond chunks, only slot 0 is real; the rest is zero-padded
    # and will be blocked by -inf in the uncond attention bias.
    S_background = background_cond_batched.shape[1] if background_cond_batched is not None else S_base
    S_total = S_background + sum(region_lengths)
    text_lengths = [S_background] + region_lengths  # slot 0 = base/background

    context_chunks = context.chunk(num_chunks, dim=0)
    unified_chunks: list[torch.Tensor] = []
    for chunk, cond_or_uncond in zip(context_chunks, cond_or_unconds):
        if cond_or_uncond == 1:
            # Uncond: pad to S_total with zeros (bias will block padded columns)
            uncond_base = _match_context_length(chunk, S_background)
            pad = torch.zeros(
                batch_size, S_total - S_background, context.shape[2],
                device=device, dtype=dtype,
            )
            unified_chunks.append(torch.cat([uncond_base, pad], dim=1))
        else:
            # Cond: append region conditionings after base/background text.
            # If background_conditioning is connected, slot 0 becomes the
            # background prompt while the original base prompt is preserved by
            # base_ratio's unpatched pass.
            base_chunk = background_cond_batched if background_cond_batched is not None else chunk
            base_chunk = _match_context_length(base_chunk, S_background)
            unified_chunks.append(torch.cat([base_chunk] + region_conds_batched, dim=1))

    unified_context = torch.cat(unified_chunks, dim=0)  # [B_total, S_total, D]

    # ---- Token masks ----------------------------------------------------
    # Compute padded token-grid dimensions (matches what _forward sees after
    # pad_to_patch_size).
    padded_t = math.ceil(latent_t / patch_temporal) * patch_temporal
    temporal_tokens = padded_t // patch_temporal
    # h_tokens / w_tokens are derived inside _masks_to_token_masks
    # to keep that function self-contained.

    # Build base mask: region that the base text covers.
    # We compute it at latent pixel resolution, then hand it to the token-mask
    # function alongside region masks (which are at their original resolution).
    padded_h = math.ceil(latent_h / patch_spatial) * patch_spatial
    padded_w = math.ceil(latent_w / patch_spatial) * patch_spatial
    region_masks_at_latent: list[torch.Tensor] = [
        F.interpolate(
            rm.unsqueeze(1), size=(padded_h, padded_w), mode="nearest-exact"
        ).squeeze(1)
        for rm in patch.region_masks
    ]

    if patch.base_mode == "global":
        base_mask = torch.ones(1, padded_h, padded_w)
    elif patch.base_mode == "disabled":
        base_mask = torch.zeros(1, padded_h, padded_w)
    else:  # uncovered_only
        base_mask = torch.ones(1, padded_h, padded_w)
        for rm in region_masks_at_latent:
            base_mask = (base_mask - rm).clamp(min=0.0)

    base_mask = base_mask * patch.base_strength

    # Slot 0 = base, slots 1..N = regions
    all_masks = [base_mask] + patch.region_masks   # original region mask resolutions are fine
    token_masks = _masks_to_token_masks(
        all_masks, latent_h, latent_w, patch_spatial, temporal_tokens
    )
    slot_strengths = torch.tensor(
        [patch.base_strength] + patch.region_weights,
        device=device,
        dtype=dtype,
    ).clamp(0.0, 1.0)

    # ---- Build attention biases ----------------------------------------
    # Cond: block-diagonal — each token attends only to its assigned slot
    # (plus base text when base_mode="global").
    cond_bias = _build_flux_cross_attention_bias(
        token_masks,
        text_lengths,
        patch.base_mode,
        device,
        dtype,
        mask_strength=patch.cross_mask_strength,
        slot_strengths=slot_strengths,
    )

    uncond_bias = torch.full(
        (1, 1, cond_bias.shape[2], S_total), float("-inf"), device=device, dtype=dtype
    )
    uncond_bias[:, :, :, :S_background] = 0.0

    bias_parts: list[torch.Tensor] = []
    for cond_or_uncond in cond_or_unconds:
        b = uncond_bias if cond_or_uncond == 1 else cond_bias
        bias_parts.append(b.expand(batch_size, -1, -1, -1))
    full_bias = torch.cat(bias_parts, dim=0)

    full_self_bias: Optional[torch.Tensor] = None
    if patch.self_mask_strength > 0.0:
        cond_self_bias = _build_flux_self_attention_bias(
            token_masks,
            patch.base_mode,
            patch.self_mask_strength,
            device,
            dtype,
            slot_strengths=slot_strengths,
        )
        uncond_self_bias = torch.zeros_like(cond_self_bias)
        self_parts: list[torch.Tensor] = []
        for cond_or_uncond in cond_or_unconds:
            b = uncond_self_bias if cond_or_uncond == 1 else cond_self_bias
            self_parts.append(b.expand(batch_size, -1, -1, -1))
        full_self_bias = torch.cat(self_parts, dim=0)

    base_output = executor(*args, **kwargs) if patch.base_ratio > 0.0 else None

    # ---- Patch cross_attn.attn_op on all blocks ------------------------
    # We replace the attn_op callable (normally torch_attention_op) with our
    # masked variant.  The original is restored in the finally block so no
    # state leaks between inference steps.
    patched: list[tuple] = []
    try:
        for block_index, block in enumerate(getattr(diffusion_model, "blocks", [])):
            if (
                patch.cross_mask_strength > 0.0
                and block_index % patch.cross_inject_every_n_blocks == 0
            ):
                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is not None:
                    original_op = cross_attn.attn_op
                    # partial binds attn_bias; call signature stays (q, k, v, transformer_options=...)
                    cross_attn.attn_op = partial(_masked_attn_op, attn_bias=full_bias)
                    patched.append((cross_attn, original_op))

            if (
                full_self_bias is not None
                and block_index % patch.self_inject_every_n_blocks == 0
            ):
                self_attn = getattr(block, "self_attn", None)
                if self_attn is not None:
                    original_op = self_attn.attn_op
                    self_attn.attn_op = partial(_masked_attn_op, attn_bias=full_self_bias)
                    patched.append((self_attn, original_op))

        # Inject unified context
        if patch.cross_mask_strength > 0.0:
            args = list(args)
            if len(args) > 2:
                args[2] = unified_context
            else:
                kwargs["context"] = unified_context
            args = tuple(args)

        regional_output = executor(*args, **kwargs)
        if base_output is not None and torch.is_tensor(regional_output) and torch.is_tensor(base_output):
            return regional_output * (1.0 - patch.base_ratio) + base_output * patch.base_ratio
        return regional_output

    finally:
        for attn, original_op in patched:
            attn.attn_op = original_op


# ---------------------------------------------------------------------------
# ComfyUI nodes
# ---------------------------------------------------------------------------

class AnimaConditioningRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "conditioning": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "regions": (REGION_TYPE,),
            },
        }

    RETURN_TYPES = (REGION_TYPE,)
    RETURN_NAMES = ("regions",)
    FUNCTION = "create"
    CATEGORY = "conditioning/Anima Regional Conditioning"

    def create(self, mask, conditioning, weight, regions=None):
        return (
            AnimaConditioningRegionChain(regions, _prepare_mask(mask), conditioning, float(weight)),
        )


class ApplyAnimaRegionalConditioningPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regions": (REGION_TYPE,),
                "base_mode": (
                    ["uncovered_only", "global", "disabled"],
                    {"default": "uncovered_only"},
                ),
                "base_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "cross_mask_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "self_mask_strength": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "base_ratio": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "cross_inject_every_n_blocks": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
                "self_inject_every_n_blocks": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
            },
            "optional": {
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "background_conditioning": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/Anima Regional Conditioning"

    def apply(
        self,
        model,
        regions,
        base_mode,
        base_strength,
        end_percent,
        cross_mask_strength,
        self_mask_strength,
        base_ratio,
        cross_inject_every_n_blocks,
        self_inject_every_n_blocks,
        start_percent=0.0,
        background_conditioning=None,
    ):
        _validate_anima_model(model)
        model_sampling = model.get_model_object("model_sampling")
        start_sigma = float(model_sampling.percent_to_sigma(start_percent))
        end_sigma = float(model_sampling.percent_to_sigma(end_percent))
        patch = AnimaRegionalConditioningPatch(
            regions.flatten(),
            base_mode,
            base_strength,
            start_sigma,
            end_sigma,
            cross_mask_strength,
            self_mask_strength,
            base_ratio,
            cross_inject_every_n_blocks,
            self_inject_every_n_blocks,
            background_conditioning,
        )

        patched_model = model.clone()
        patched_model.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY
        )
        patched_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            _diffusion_model_wrapper,
        )
        patched_model.model_options.setdefault("transformer_options", {})[WRAPPER_KEY] = patch
        patched_model.set_attachments(WRAPPER_KEY, patch)
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "AnimaConditioningRegion": AnimaConditioningRegion,
    "ApplyAnimaRegionalConditioningPatch": ApplyAnimaRegionalConditioningPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaConditioningRegion": "Anima Conditioning Region",
    "ApplyAnimaRegionalConditioningPatch": "Apply Anima Regional Conditioning Patch",
}
