from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_yaml(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _names_to_dict(names) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def resolve_split_path(data: Dict, split: str) -> str:
    if split not in data:
        raise KeyError(f"'{split}' not found in data yaml.")

    base = data.get('path', '')
    split_path = data[split]
    if os.path.isabs(split_path):
        return split_path
    return str(Path(base) / split_path) if base else split_path


def list_images(img_path: str) -> List[str]:
    p = Path(img_path)
    files: List[str] = []

    if p.is_dir():
        for ext in IMG_EXTS:
            files.extend(str(x) for x in p.rglob(f'*{ext}'))
            files.extend(str(x) for x in p.rglob(f'*{ext.upper()}'))
    elif p.is_file():
        with open(p, 'r', encoding='utf-8') as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        parent = p.parent
        for x in lines:
            files.append(x if os.path.isabs(x) else str((parent / x).resolve()))
    else:
        raise FileNotFoundError(f'Image path not found: {img_path}')

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f'No images found in: {img_path}')
    return files


def img2label_path(im_file: str) -> str:
    p = Path(im_file)
    parts = list(p.parts)
    if 'images' in parts:
        parts[parts.index('images')] = 'labels'
        return str(Path(*parts).with_suffix('.txt'))
    return str(p.with_suffix('.txt'))


def load_yolo_label(label_file: str) -> np.ndarray:
    if not os.path.exists(label_file):
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 5:
                continue
            rows.append([float(x) for x in s[:5]])

    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def xywhn2xyxy(x: np.ndarray, w: int, h: int) -> np.ndarray:
    y = x.copy()
    y[:, 0] = (x[:, 0] - x[:, 2] / 2) * w
    y[:, 1] = (x[:, 1] - x[:, 3] / 2) * h
    y[:, 2] = (x[:, 0] + x[:, 2] / 2) * w
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) * h
    return y


def xyxy2xywhn(x: np.ndarray, w: int, h: int, eps: float = 1e-6) -> np.ndarray:
    y = x.copy()
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / max(w, eps)
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / max(h, eps)
    y[:, 2] = (x[:, 2] - x[:, 0]) / max(w, eps)
    y[:, 3] = (x[:, 3] - x[:, 1]) / max(h, eps)
    return y


def clip_boxes_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def letterbox(
    image: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, (r, r), (dw, dh)


class YOLOLikeDetectionDataset(Dataset):
    def __init__(
        self,
        img_path: str,
        imgsz: int = 640,
        augment: bool = False,
        cache_images: bool = False,
        hflip: float = 0.5,
    ):
        super().__init__()
        self.imgsz = imgsz
        self.augment = augment
        self.cache_images = cache_images
        self.hflip = hflip

        self.im_files = list_images(img_path)
        self.label_files = [img2label_path(x) for x in self.im_files]
        self.cache = {}
        if self.cache_images:
            for im_file in self.im_files:
                img = cv2.imread(im_file)
                if img is None:
                    raise FileNotFoundError(f'Failed to read image: {im_file}')
                self.cache[im_file] = img

    def __len__(self) -> int:
        return len(self.im_files)

    def _read_image(self, im_file: str) -> np.ndarray:
        if self.cache_images and im_file in self.cache:
            return self.cache[im_file].copy()
        img = cv2.imread(im_file)
        if img is None:
            raise FileNotFoundError(f'Failed to read image: {im_file}')
        return img

    def __getitem__(self, index: int) -> Dict:
        im_file = self.im_files[index]
        lb_file = self.label_files[index]

        img = self._read_image(im_file)
        h0, w0 = img.shape[:2]

        labels = load_yolo_label(lb_file)
        if len(labels):
            cls = labels[:, 0:1]
            bboxes = xywhn2xyxy(labels[:, 1:5], w0, h0)
        else:
            cls = np.zeros((0, 1), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)

        img, ratio, pad = letterbox(img, new_shape=self.imgsz, scaleup=True)
        h1, w1 = img.shape[:2]

        if len(bboxes):
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * ratio[0] + pad[0]
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * ratio[1] + pad[1]

        if self.augment and random.random() < self.hflip:
            img = np.ascontiguousarray(img[:, ::-1, :])
            if len(bboxes):
                x1 = bboxes[:, 0].copy()
                x2 = bboxes[:, 2].copy()
                bboxes[:, 0] = w1 - x2
                bboxes[:, 2] = w1 - x1

        if len(bboxes):
            bboxes = clip_boxes_xyxy(bboxes, w1, h1)
            bboxes = xyxy2xywhn(bboxes, w1, h1).astype(np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        return {
            'im_file': im_file,
            'ori_shape': (h0, w0),
            'resized_shape': (h1, w1),
            'ratio_pad': (ratio, pad),
            'img': torch.from_numpy(img),
            'cls': torch.from_numpy(cls.astype(np.float32)),
            'bboxes': torch.from_numpy(bboxes.astype(np.float32)),
            'batch_idx': torch.zeros((len(cls),), dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        new_batch: Dict = {}
        keys = batch[0].keys()
        for k in keys:
            vals = [b[k] for b in batch]
            if k == 'img':
                new_batch[k] = torch.stack(vals, 0)
            elif k in ['cls', 'bboxes']:
                new_batch[k] = torch.cat(vals, 0) if vals else torch.empty(0)
            elif k == 'batch_idx':
                batch_idx = []
                for i, v in enumerate(vals):
                    batch_idx.append(v + i)
                new_batch[k] = torch.cat(batch_idx, 0) if batch_idx else torch.empty(0)
            else:
                new_batch[k] = vals
        return new_batch


def build_loader(
    data_yaml: str,
    split: str,
    imgsz: int,
    batch_size: int,
    workers: int,
    cache_images: bool = False,
    hflip: float = 0.5,
):
    data = load_yaml(data_yaml)
    img_path = resolve_split_path(data, split=split)
    dataset = YOLOLikeDetectionDataset(
        img_path=img_path,
        imgsz=imgsz,
        augment=(split == 'train'),
        cache_images=cache_images,
        hflip=hflip,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(len(dataset), 1)),
        shuffle=(split == 'train'),
        num_workers=workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    meta = {
        'nc': int(data.get('nc', len(_names_to_dict(data.get('names', {}))))),
        'names': _names_to_dict(data.get('names', {})),
    }
    return loader, meta


def build_train_loader(data_yaml: str, imgsz: int = 640, batch_size: int = 4, workers: int = 0,
                       cache_images: bool = False, hflip: float = 0.5):
    return build_loader(data_yaml, 'train', imgsz, batch_size, workers, cache_images=cache_images, hflip=hflip)


def build_val_loader(data_yaml: str, imgsz: int = 640, batch_size: int = 4, workers: int = 0,
                     cache_images: bool = False):
    return build_loader(data_yaml, 'val', imgsz, batch_size, workers, cache_images=cache_images, hflip=0.0)
