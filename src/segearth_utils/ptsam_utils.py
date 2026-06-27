"""PTSAM soft-prompt injection utilities.

Injects learnable soft tokens into SAM3's language_features before the mask
decoder cross-attention. Only these tokens are trainable; the entire backbone
(PE-L+ image encoder + text encoder) stays frozen.

Architecture reference:
    language_features: [seq_len, batch, 256]   (from backbone.forward_text)
    language_mask:     [batch, seq_len]
    Soft tokens concat along seq_len (dim 0 / dim 1 respectively).

PTSAM paper: "Prompt-Tuning SAM: Specialist with 2048 params, 16 images"
             CVPRW 2025. We use 8 tokens × 256 = 2048 params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_SOFT_TOKENS = 8          # 8 × 256 = 2048 trainable params
SOFT_EMBED_DIM = 256       # matches SAM3 language_features dim


def make_soft_prompts(device="cuda", n_tokens=N_SOFT_TOKENS, dim=SOFT_EMBED_DIM):
    """Create the learnable soft-prompt tensor (the ONLY trainable parameter)."""
    return nn.Parameter(torch.randn(n_tokens, 1, dim, device=device) * 0.02)


def inject_soft_prompts(backbone_out: dict, soft_prompts: nn.Parameter) -> dict:
    """Return a shallow-copy of backbone_out with soft tokens appended.

    Keeps the original tensors detached (no grad through frozen backbone).
    Only soft_prompts contribute gradients.

    Args:
        backbone_out: dict with "language_features" [S, B, 256] and
                      "language_mask" [B, S], already populated by
                      model.backbone.forward_text.
        soft_prompts: nn.Parameter [N, 1, 256]

    Returns:
        New dict with features/mask extended by soft tokens.
    """
    out = dict(backbone_out)  # shallow copy — don't mutate caller's dict

    lang_feat = out["language_features"].detach()   # [S, B, 256]
    lang_mask = out["language_mask"].detach()        # [B, S]

    B = lang_feat.shape[1]
    soft = soft_prompts.expand(-1, B, -1)           # [N, B, 256]
    soft_mask = torch.ones(B, soft_prompts.shape[0],
                           device=lang_feat.device,
                           dtype=lang_mask.dtype)   # [B, N]

    out["language_features"] = torch.cat([lang_feat, soft], dim=0)   # [S+N, B, 256]
    out["language_mask"] = torch.cat([lang_mask, soft_mask], dim=1)  # [B, S+N]
    return out


def build_logit_map(model, cached_backbone_out: dict, soft_prompts: nn.Parameter,
                    query_words: list, find_stage, patch_hw: tuple,
                    device="cuda") -> torch.Tensor:
    """Run decoder for all classes and return stacked logit map.

    Args:
        model: SAM3 model object (all params frozen except soft_prompts)
        cached_backbone_out: dict returned by model.backbone.forward_image,
                             WITHOUT language_features (those are added here)
        soft_prompts: nn.Parameter [N, 1, 256]
        query_words: list of str, one per class (first synonym)
        find_stage: FindStage instance (shared, re-used)
        patch_hw: (H, W) of the original image/patch

    Returns:
        all_logits: float32 tensor [N_CLS, H, W] (not yet argmaxed)
    """
    H, W = patch_hw
    dummy_geom = model._get_dummy_prompt()
    all_logits = []

    for query_word in query_words:
        with torch.no_grad():
            text_out = model.backbone.forward_text([query_word], device=device)

        cur_backbone = {**cached_backbone_out, **text_out}
        cur_backbone = inject_soft_prompts(cur_backbone, soft_prompts)

        outputs = model.forward_grounding(
            backbone_out=cur_backbone,
            find_input=find_stage,
            geometric_prompt=dummy_geom,
            find_target=None,
        )

        sem_logit = outputs["semantic_seg"]                     # [1, 1, H', W']
        sem_logit = F.interpolate(sem_logit, size=(H, W),
                                  mode="bilinear", align_corners=False)
        presence = outputs["presence_logit_dec"].sigmoid()
        all_logits.append(sem_logit.squeeze() * presence.squeeze())

    return torch.stack(all_logits, dim=0)   # [N_CLS, H, W]


# ── Label conversion helpers ──────────────────────────────────────────────────

RGB_TO_IDX_1INDEXED = {
    (255, 255, 255): 1,   # impervious surface
    (  0,   0, 255): 2,   # building
    (  0, 255, 255): 3,   # low vegetation
    (  0, 255,   0): 4,   # tree
    (255, 255,   0): 5,   # car
    (255,   0,   0): 6,   # clutter
}


def rgb_label_to_gt(label_rgb_arr) -> torch.Tensor:
    """Convert Potsdam RGB label array to 0-indexed GT tensor.

    Applies reduce_zero_label manually (1-indexed → 0-indexed).
    Pixels not in RGB_TO_IDX → 255 (ignored).

    Returns:
        int64 tensor [H, W], values 0-5 or 255
    """
    import numpy as np
    h, w = label_rgb_arr.shape[:2]
    gt = torch.full((h, w), 255, dtype=torch.int64)
    arr = label_rgb_arr[:, :, :3] if label_rgb_arr.ndim == 3 else label_rgb_arr
    for (r, g, b), idx in RGB_TO_IDX_1INDEXED.items():
        mask = (arr[:, :, 0] == r) & (arr[:, :, 1] == g) & (arr[:, :, 2] == b)
        gt[mask] = idx - 1  # reduce_zero_label: 1→0, ..., 6→5
    return gt
