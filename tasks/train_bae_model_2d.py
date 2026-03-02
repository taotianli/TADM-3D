"""
Training script for BAE2D (2D version of Brain Age Estimator).
"""
import sys
sys.path.append('.')
sys.path.append('./models')

import os
import argparse
import numpy as np

import monai
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from tqdm import tqdm

from models.BAE2D import BAE2D
from utilities.lr_scheduler import LinearWarmupCosineAnnealingLR
from utilities import utils, const

import wandb

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_slice(vol, axis):
    """Extract centre slice (B, C, D, H, W) → (B, C, H, W)."""
    mid = vol.shape[axis + 2] // 2
    if axis == 0:
        return vol[:, :, mid, :, :]
    elif axis == 1:
        return vol[:, :, :, mid, :]
    else:
        return vol[:, :, :, :, mid]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     required=True,  type=str)
    parser.add_argument('--cache_dir',   required=True,  type=str)
    parser.add_argument('--output_dir',  required=True,  type=str)
    parser.add_argument('--num_workers', default=8,      type=int)
    parser.add_argument('--n_epochs',    default=500,    type=int)
    parser.add_argument('--batch_size',  default=16,     type=int)
    parser.add_argument('--lr',          default=1e-4,   type=float)
    parser.add_argument('--wandb',       action='store_true')
    parser.add_argument('--run_name',    required=True,  type=str)
    parser.add_argument('--slice_axis',  default=2,      type=int,
                        help='Axis for 2D slice extraction (0=sagittal,1=coronal,2=axial)')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="TADM-2D-BAE", name=args.run_name)

    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['img_hr', 'img_lr'], reader="NibabelReader"),
        transforms.EnsureChannelFirstD(keys=['img_lr', 'img_hr']),
        transforms.SpacingD(pixdim=1.5, keys=['img_lr', 'img_hr']),
        transforms.ResizeWithPadOrCropD(spatial_size=(128, 128, 128),
                                        mode='minimum', keys=['img_lr', 'img_hr'], lazy=True),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['img_hr', 'img_lr']),
    ])

    train_df = pd.read_csv(args.dataset + "train_dataset.csv").to_dict(orient='records')
    valid_df = pd.read_csv(args.dataset + "valid_dataset.csv").to_dict(orient='records')
    trainset = monai.data.Dataset(train_df, transforms_fn)
    validset = monai.data.Dataset(valid_df, transforms_fn)

    train_loader = DataLoader(trainset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=True,
                              persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(validset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False,
                              persistent_workers=True, pin_memory=True)

    bae = BAE2D().to(DEVICE)
    optimizer = torch.optim.AdamW(bae.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=500)
    scaler    = GradScaler()
    writer    = SummaryWriter()

    global_counter = {'train': 0, 'valid': 0}
    loaders        = {'train': train_loader, 'valid': valid_loader}
    min_valid_loss = np.inf

    for epoch in range(args.n_epochs):
        for mode in loaders:
            loader = loaders[mode]
            bae.train() if mode == 'train' else bae.eval()

            epoch_loss = 0.0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch} [{mode}]")

            for step, batch in progress_bar:
                with autocast(enabled=True):
                    if mode == 'train':
                        optimizer.zero_grad(set_to_none=True)

                    hr_2d = extract_slice(batch['img_hr'], args.slice_axis).to(DEVICE)
                    lr_2d = extract_slice(batch['img_lr'], args.slice_axis).to(DEVICE)

                    age               = batch['age'].to(DEVICE)
                    patient_condition = batch['patient_condition'].to(DEVICE)
                    diff_ages         = batch['diff_ages'].to(DEVICE)

                    with torch.set_grad_enabled(mode == 'train'):
                        pred_diff_age = bae(lr_2d, hr_2d, age, patient_condition)
                        loss = F.mse_loss(pred_diff_age.float(), diff_ages.float().unsqueeze(1))

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1

            epoch_loss /= len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            if args.wandb:
                wandb.log({f'{mode}/epoch-mse': epoch_loss}, step=epoch)

        scheduler.step()

        if epoch_loss < min_valid_loss:
            min_valid_loss = epoch_loss
            torch.save(bae.state_dict(),
                       os.path.join(args.output_dir, 'bae2d-best.pth'))

        torch.save(bae.state_dict(),
                   os.path.join(args.output_dir, 'bae2d-last.pth'))
