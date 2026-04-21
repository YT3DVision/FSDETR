from __future__ import annotations

from typing import Dict, Tuple

import torch

from FSDETR.losses.loss import RTDETRDetectionLoss


def build_criterion(num_classes: int) -> RTDETRDetectionLoss:
    return RTDETRDetectionLoss(
        nc=num_classes,
        use_vfl=True,
        use_sl=False,
        use_emasl=False,
        use_svfl=False,
        use_emasvfl=False,
        use_mal=False,
    )


def build_targets(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    imgs = batch["img"].to(device, non_blocking=True)
    batch_idx = batch["batch_idx"].to(device).long().view(-1)

    targets = {
        "cls": batch["cls"].to(device).long().view(-1),
        "bboxes": batch["bboxes"].to(device),
        "batch_idx": batch_idx,
        "gt_groups": [(batch_idx == i).sum().item() for i in range(imgs.shape[0])],
    }
    return imgs, targets


def compute_rtdetr_loss(model, criterion, batch: Dict, device: torch.device):
    imgs, targets = build_targets(batch, device)

    preds = model(imgs, batch=targets, return_raw=True)
    dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds

    if dn_meta is None:
        dn_bboxes, dn_scores = None, None
    else:
        dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
        dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

    dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes], dim=0)
    dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores], dim=0)

    loss_dict = criterion(
        (dec_bboxes, dec_scores),
        targets,
        dn_bboxes=dn_bboxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
    )
    total_loss = sum(loss_dict.values())

    log_items = {k: float(v.detach().item()) for k, v in loss_dict.items()}
    return total_loss, log_items