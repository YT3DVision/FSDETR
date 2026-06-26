from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import FSDETRModel
from utils.common import build_criterion, compute_rtdetr_loss
from utilis.my_data import build_train_loader, build_val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FSDETR")
    parser.add_argument('--data', type=str, required=True, help='Path to data yaml.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0, help='Use 0 on Windows if dataloader is unstable.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-classes', type=int, default=None, help='Override number of classes. If omitted, read from data yaml.')
    parser.add_argument('--save-dir', type=str, default='runs/train/exp')
    parser.add_argument('--val-interval', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--cache-images', action='store_true')
    parser.add_argument('--hflip', type=float, default=0.5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--clip-grad', type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_val(
    model: FSDETRModel,
    criterion,
    val_loader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    meter: Dict[str, float] = {}

    for batch in val_loader:
        loss, log_items = compute_rtdetr_loss(model, criterion, batch, device)
        total_loss += float(loss.item())
        count += 1
        for k, v in log_items.items():
            meter[k] = meter.get(k, 0.0) + float(v)

    out = {'loss': total_loss / max(count, 1)}
    for k, v in meter.items():
        out[k] = v / max(count, 1)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, meta = build_train_loader(
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        cache_images=args.cache_images,
        hflip=args.hflip,
    )
    val_loader, _ = build_val_loader(
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        cache_images=args.cache_images,
    )

    num_classes = args.num_classes or meta['nc']
    model = FSDETRModel(num_classes=num_classes).to(device)
    criterion = build_criterion(num_classes).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    loss, log_items = compute_rtdetr_loss(model, criterion, batch, device)
                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, log_items = compute_rtdetr_loss(model, criterion, batch, device)
                loss.backward()
                if args.clip_grad > 0:
                    clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                optimizer.step()

            running_loss += float(loss.item())
            step_count += 1

            if step % args.print_freq == 0:
                print(
                    f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                    f"[Step {step:04d}/{len(train_loader):04d}] "
                    f"loss={loss.item():.4f} "
                    f"giou={log_items.get('loss_giou', 0.0):.4f} "
                    f"cls={log_items.get('loss_class', 0.0):.4f} "
                    f"bbox={log_items.get('loss_bbox', 0.0):.4f}"
                )

        scheduler.step()
        train_loss = running_loss / max(step_count, 1)

        metrics = {'loss': float('nan')}
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            metrics = run_val(model, criterion, val_loader, device)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val': best_val,
            'args': vars(args),
            'num_classes': num_classes,
            'names': meta.get('names', {}),
        }

        torch.save(ckpt, save_dir / 'last.pth')
        if epoch % args.save_interval == 0:
            torch.save(ckpt, save_dir / f'epoch_{epoch:03d}.pth')

        current_val = metrics['loss']
        if current_val < best_val:
            best_val = current_val
            ckpt['best_val'] = best_val
            torch.save(ckpt, save_dir / 'best.pth')

        print(
            f"Epoch {epoch:03d} completed | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={current_val:.4f} | "
            f"best_val={best_val:.4f}"
        )


if __name__ == '__main__':
    main()
