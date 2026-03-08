"""
Training script for Diffusion2D (2D version of TADM-3D).
"""
import sys
sys.path.append('.')
sys.path.append('./models')

import os
import argparse
import random
import numpy as np

import monai
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from tqdm import tqdm
import psutil

from models.Diffusion2D import Diffusion2D
from models.BAE2D import BAE2D
from utilities.lr_scheduler import LinearWarmupCosineAnnealingLR
from utilities import utils, const

import wandb

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def norm_output(output):
    return (output - output.min()) / (output.max() - output.min() + 1e-8)


def images_sampling(image, metadata, diffusion):
    with torch.no_grad():
        output = diffusion(image=image, metadata=metadata, pred_type="ddim_sample")
        return torch.clamp(output, -1, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     required=True,  type=str)
    parser.add_argument('--cache_dir',   required=True,  type=str)
    parser.add_argument('--output_dir',  required=True,  type=str)
    parser.add_argument('--diff_ckpt',   default=None,   type=str)
    parser.add_argument('--num_workers', default=8,      type=int)
    parser.add_argument('--n_epochs',    default=500,    type=int)
    parser.add_argument('--batch_size',  default=16,     type=int)
    parser.add_argument('--lr',          default=1e-4,   type=float)
    parser.add_argument('--wandb',       action='store_true')
    parser.add_argument('--run_name',    required=True,  type=str)
    parser.add_argument('--bae_ckpt',    default=None,   type=str)
    # 2D-specific: which slice axis and index to extract
    parser.add_argument('--slice_axis',  default=2,      type=int,
                        help='Axis along which to extract 2D slice (0=sagittal,1=coronal,2=axial)')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="TADM-2D-Diff", name=args.run_name)

    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['img_hr', 'img_lr'], reader="NibabelReader"),
        transforms.EnsureChannelFirstD(keys=['img_lr', 'img_hr']),
        transforms.SpacingD(pixdim=1.5, keys=['img_lr', 'img_hr']),
        transforms.ResizeWithPadOrCropD(spatial_size=(128, 128, 128),
                                        mode='minimum', keys=['img_lr', 'img_hr'], lazy=True),
        transforms.NormalizeIntensityD(keys=['img_hr', 'img_lr'], nonzero=True, channel_wise=True),
    ])

    def load_df(csv_path, dataset_root):
        df = pd.read_csv(csv_path)
        for col in ('img_hr', 'img_lr'):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda p: p if os.path.isabs(p) else os.path.join(dataset_root, p)
                )
        return df.to_dict(orient='records')

    train_df = load_df(args.dataset + "train_dataset.csv", args.dataset)
    valid_df = load_df(args.dataset + "valid_dataset.csv", args.dataset)
    trainset = monai.data.Dataset(train_df, transforms_fn)
    validset = monai.data.Dataset(valid_df, transforms_fn)

    train_loader = DataLoader(trainset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=True,
                              persistent_workers=args.num_workers > 0, pin_memory=False)
    valid_loader = DataLoader(validset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False,
                              persistent_workers=args.num_workers > 0, pin_memory=False)

    diffusion = Diffusion2D().to(DEVICE)

    if args.diff_ckpt is not None:
        diffusion.load_state_dict(torch.load(args.diff_ckpt, map_location=DEVICE))
        print(f"Loaded checkpoint from {args.diff_ckpt}")

    bae = None
    if args.bae_ckpt is not None:
        bae = BAE2D().to(DEVICE)
        bae.load_state_dict(torch.load(args.bae_ckpt, map_location=DEVICE))
        bae.eval()
        for p in bae.parameters():
            p.requires_grad = False
        print(f"Loaded and froze BAE2D from {args.bae_ckpt}")

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=500)
    scaler    = GradScaler('cuda')
    writer    = SummaryWriter()

    global_counter = {'train': 0, 'valid': 0}
    loaders        = {'train': train_loader, 'valid': valid_loader}
    min_valid_loss = np.inf

    def extract_slice(vol, axis):
        """Extract centre slice from a 3D volume (B, C, D, H, W) → (B, C, H, W)."""
        mid = vol.shape[axis + 2] // 2
        if axis == 0:
            return vol[:, :, mid, :, :]
        elif axis == 1:
            return vol[:, :, :, mid, :]
        else:
            return vol[:, :, :, :, mid]

    for epoch in range(args.n_epochs):
        for mode in loaders:
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()

            epoch_loss = epoch_diff_loss = epoch_bae_loss = 0.0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch} [{mode}]")

            for step, batch in progress_bar:
                with autocast('cuda', enabled=True):
                    if mode == 'train':
                        optimizer.zero_grad(set_to_none=True)

                    reverse = (random.randint(0, 1) == 1) and (mode == 'train')

                    # Extract 2D slices
                    img_hr_2d = extract_slice(batch['img_hr'], args.slice_axis).to(DEVICE)
                    img_lr_2d = extract_slice(batch['img_lr'], args.slice_axis).to(DEVICE)

                    diff_ages_yr = batch['diff_ages'] / 12.0

                    if not reverse:
                        inputs   = img_hr_2d - img_lr_2d
                        metadata = torch.stack(
                            (batch['age'], diff_ages_yr, batch['patient_condition']), dim=1
                        ).float().to(DEVICE)
                        context  = img_lr_2d
                    else:
                        inputs   = img_lr_2d - img_hr_2d
                        metadata = torch.stack(
                            (batch['age'] + diff_ages_yr,
                             -diff_ages_yr,
                             batch['patient_condition']), dim=1
                        ).float().to(DEVICE)
                        context  = img_hr_2d

                    with torch.set_grad_enabled(mode == 'train'):
                        x_t, t, noise = diffusion(x=inputs, pred_type="q_sample")
                        pred_inputs   = diffusion(x=x_t, step=t, image=context,
                                                  metadata=metadata, pred_type="denoise")

                        diff_loss = F.mse_loss(pred_inputs.float(), inputs.float())
                        loss      = diff_loss

                        bae_loss = torch.tensor(0.0, device=DEVICE)
                        if bae is not None:
                            if not reverse:
                                pred_hr = context + pred_inputs
                                predicted_diff_age = bae(
                                    context, pred_hr,
                                    batch['age'].to(DEVICE),
                                    batch['patient_condition'].to(DEVICE))
                            else:
                                pred_lr = context - pred_inputs
                                predicted_diff_age = bae(
                                    pred_lr, context,
                                    batch['age'].to(DEVICE),
                                    batch['patient_condition'].to(DEVICE))
                            gt_diff_age = (batch['diff_ages'] / 12.0).to(DEVICE).unsqueeze(1).float()
                            bae_loss = F.mse_loss(predicted_diff_age.float(), gt_diff_age)
                            loss = diff_loss + bae_loss

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-loss',      loss.item(),      global_counter[mode])
                writer.add_scalar(f'{mode}/batch-diff-loss', diff_loss.item(), global_counter[mode])
                if bae is not None:
                    writer.add_scalar(f'{mode}/batch-bae-loss', bae_loss.item(), global_counter[mode])

                epoch_loss      += loss.item()
                epoch_diff_loss += diff_loss.item()
                epoch_bae_loss  += bae_loss.item()
                gpu_alloc   = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                ram_gb      = psutil.Process().memory_info().rss / 1024**3
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss / (step + 1):.4f}",
                    "GPU":  f"{gpu_alloc:.1f}/{gpu_reserved:.1f}GB",
                    "RAM":  f"{ram_gb:.1f}GB",
                })
                global_counter[mode] += 1

            scheduler.step()
            n = len(loader)
            epoch_loss      /= n
            epoch_diff_loss /= n
            epoch_bae_loss  /= n

            writer.add_scalar(f'{mode}/epoch-loss',      epoch_loss,      epoch)
            writer.add_scalar(f'{mode}/epoch-diff-loss', epoch_diff_loss, epoch)
            if bae is not None:
                writer.add_scalar(f'{mode}/epoch-bae-loss', epoch_bae_loss, epoch)

            diff_pred = images_sampling(context[0].unsqueeze(0), metadata[0].unsqueeze(0), diffusion)
            pred_image = torch.clamp(context[0] + diff_pred[0], 0, 1)

            if args.wandb:
                log_dict = {
                    f'{mode}/epoch-loss':    epoch_loss,
                    f'{mode}/epoch-diff':    epoch_diff_loss,
                    f'{mode}/predicted_hr':  wandb.Image(pred_image[0].detach().cpu()),
                    f'{mode}/img_hr':        wandb.Image(img_hr_2d[0][0].detach().cpu()),
                    f'{mode}/img_lr':        wandb.Image(img_lr_2d[0][0].detach().cpu()),
                }
                if bae is not None:
                    log_dict[f'{mode}/epoch-bae'] = epoch_bae_loss
                wandb.log(log_dict, step=epoch)

        if epoch_loss < min_valid_loss:
            min_valid_loss = epoch_loss
            torch.save(diffusion.state_dict(),
                       os.path.join(args.output_dir, 'diff2d-best.pth'))

        torch.save(diffusion.state_dict(),
                   os.path.join(args.output_dir, 'diff2d-last.pth'))
