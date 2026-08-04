"""Pascal VOC semantic segmentation transfer for the rebuttal."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from experiment.models.backbones.Resnet import ResNet50


def encoder_map(model, x):
    r = model.resnet
    x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
    x = r.layer1(x); x = r.layer2(x); x = r.layer3(x)
    return r.layer4(x)


def batch_transform(batch):
    images, masks = zip(*batch)
    image_t = torch.stack([transforms.functional.resize(transforms.functional.to_tensor(x), (224, 224)) for x in images])
    image_t = transforms.functional.normalize(image_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mask_t = torch.stack([transforms.functional.resize(torch.as_tensor(np.array(m), dtype=torch.long)[None], (224, 224), interpolation=transforms.InterpolationMode.NEAREST)[0] for m in masks])
    return image_t, mask_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--finetune-encoder',
        action='store_true',
        help='Fine-tune the final ResNet stage as well as the segmentation head.',
    )
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    model = ResNet50(128)
    model.load_state_dict({k.removeprefix('model.'): v for k, v in state.items() if k.startswith('model.')}, strict=False)
    model.to(device).eval()
    for q in model.parameters(): q.requires_grad = False
    if args.finetune_encoder:
        for q in model.resnet.layer4.parameters():
            q.requires_grad = True
        model.resnet.layer4.train()
    head = nn.Conv2d(2048, 21, 1).to(device)
    groups = [{"params": head.parameters(), "lr": 1e-3}]
    if args.finetune_encoder:
        groups.append({"params": model.resnet.layer4.parameters(), "lr": 1e-5})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    train = datasets.VOCSegmentation(args.root, year='2012', image_set='train', download=True)
    val = datasets.VOCSegmentation(args.root, year='2012', image_set='val', download=True)
    for _ in range(args.epochs):
        for x, y in DataLoader(train, batch_size=16, shuffle=True, num_workers=2, collate_fn=batch_transform):
            if args.finetune_encoder:
                f = encoder_map(model, x.to(device))
            else:
                with torch.no_grad(): f = encoder_map(model, x.to(device))
            logits = F.interpolate(head(f), (224, 224), mode='bilinear', align_corners=False)
            loss = F.cross_entropy(logits, y.to(device), ignore_index=255); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    head.eval()
    inter = torch.zeros(21, device=device); union = torch.zeros(21, device=device)
    with torch.no_grad():
        for x, y in DataLoader(val, batch_size=16, num_workers=2, collate_fn=batch_transform):
            pred = F.interpolate(head(encoder_map(model, x.to(device))), (224, 224), mode='bilinear', align_corners=False).argmax(1); y=y.to(device)
            for c in range(21): inter[c] += ((pred==c)&(y==c)).sum(); union[c] += ((pred==c)|(y==c)).sum()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        f"mIoU={(inter/union.clamp_min(1)).mean().item():.6f}\n"
        f"encoder_finetuned={args.finetune_encoder}\n"
    )

if __name__ == '__main__': main()
