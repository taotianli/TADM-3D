"""
Training script for FlowMatching2D (2D version of TAFM-3D).
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

from models.FlowMatching2D import FlowMatching2D
from models.BAE2D import BAE2D
from utilities.lr_scheduler import LinearWarmupCosineAnnealingLR
from utilities import utils, const

import wandb

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def norm_output(output):
    return (output - output.min()) / (output.max() - output.min() + 1e-8)


@torch.no_grad()
def images_sampling(image, metadata, model, diff_ages=None):
    output = model(image=image, metadata=metadata,
                   pred_type="fm_sample", diff_ages=diff_ages)
    return torch.clamp(output, -1, 1)


def consistency_loss(model, x1, image, metadata, diff_ages, n_pairs=2):
    """Consistency regularization (Innovation 5) — 2D version."""
    embeddings = model.embed_model(image)
    preds = []
    for _ in range(n_pairs):
        x_t, t, _ = model(x=x1, pred_type="get_train_sample", diff_ages=diff_ages)
        v_pred = model.model(x_t, t=t, image=image,
                             embeddings=embeddings, metadata=metadata)
        t_ = t.view(-1, 1, 1, 1)
        x1_hat = x_t + (1.0 - t_) * v_pred
        preds.append(x1_hat)
    return F.mse_loss(preds[0], preds[1].detach())


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
    parser.add_argument('--fm_ckpt',     default=None,   type=str)
    parser.add_argument('--num_workers', default=8,      type=int)
    parser.add_argument('--n_epochs',    default=500,    type=int)
    parser.add_argument('--batch_size',  default=16,     type=int)
    parser.add_argument('--lr',          default=1e-4,   type=float)
    parser.add_argument('--wandb',       action='store_true')
    parser.add_argument('--run_name',    required=True,  type=str)
    parser.add_argument('--bae_ckpt',    default=None,   type=str)
    parser.add_argument('--lambda_cons', default=0.1,    type=float)
    parser.add_argument('--interpolant', default='stochastic', type=str,
                        choices=['linear', 'cosine', 'stochastic'])
    parser.add_argument('--solver',      default='heun', type=str,
                        choices=['euler', 'heun'])
    parser.add_argument('--n_steps',     default=20,     type=int)
    parser.add_argument('--slice_axis',  default=2,      type=int,
                        help='Axis for 2D slice extraction (0/1/2)')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="TAFM-2D", name=args.run_name)

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

    model = FlowMatching2D(
        interpolant_type=args.interpolant,
        n_inference_steps=args.n_steps,
        solver=args.solver,
    ).to(DEVICE)

    if args.fm_ckpt is not None:
        model.load_state_dict(torch.load(args.fm_ckpt, map_location=DEVICE))
        print(f"Loaded checkpoint from {args.fm_ckpt}")

    bae = None
    if args.bae_ckpt is not None:
        bae = BAE2D().to(DEVICE)
        bae.load_state_dict(torch.load(args.bae_ckpt, map_location=DEVICE))
        bae.eval()
        for p in bae.parameters():
            p.requires_grad = False
        print(f"Loaded and froze BAE2D from {args.bae_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=500)
    scaler    = GradScaler('cuda')
    writer    = SummaryWriter()

    global_counter = {'train': 0, 'valid': 0}
    loaders        = {'train': train_loader, 'valid': valid_loader}
    min_valid_loss = np.inf

    for epoch in range(args.n_epochs):
        model.fm.time_sampler.anneal(epoch, args.n_epochs)

        for mode in loaders:
            loader = loaders[mode]
            model.train() if mode == 'train' else model.eval()

            epoch_loss = epoch_cfm_loss = epoch_bae_loss = epoch_cons_loss = 0.0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch} [{mode}]")

            for step, batch in progress_bar:
                with autocast('cuda', enabled=True):
                    if mode == 'train':
                        optimizer.zero_grad(set_to_none=True)

                    reverse = (random.randint(0, 1) == 1) and (mode == 'train')

                    img_hr_2d = extract_slice(batch['img_hr'], args.slice_axis).to(DEVICE)
                    img_lr_2d = extract_slice(batch['img_lr'], args.slice_axis).to(DEVICE)

                    diff_ages_yr = batch['diff_ages'] / 12.0

                    if not reverse:
                        inputs    = img_hr_2d - img_lr_2d
                        metadata  = torch.stack(
                            (batch['age'], diff_ages_yr, batch['patient_condition']), dim=1
                        ).float().to(DEVICE)
                        context   = img_lr_2d
                        diff_ages = diff_ages_yr.to(DEVICE)
                    else:
                        inputs    = img_lr_2d - img_hr_2d
                        metadata  = torch.stack(
                            (batch['age'] + diff_ages_yr,
                             -diff_ages_yr,
                             batch['patient_condition']), dim=1
                        ).float().to(DEVICE)
                        context   = img_hr_2d
                        diff_ages = diff_ages_yr.to(DEVICE)

                    with torch.set_grad_enabled(mode == 'train'):
                        x_t, t, v_target = model(
                            x=inputs, pred_type="get_train_sample",
                            diff_ages=diff_ages)

                        v_pred = model(
                            x=x_t, step=t, image=context,
                            metadata=metadata, pred_type="predict_velocity")

                        cfm_loss = F.mse_loss(v_pred.float(), v_target.float())
                        loss = cfm_loss

                        bae_loss = torch.tensor(0.0, device=DEVICE)
                        if bae is not None:
                            t_ = t.view(-1, 1, 1, 1)
                            pred_inputs = x_t + (1.0 - t_) * v_pred
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
                            loss = loss + bae_loss

                        cons_loss = torch.tensor(0.0, device=DEVICE)
                        if args.lambda_cons > 0 and mode == 'train':
                            cons_loss = consistency_loss(
                                model, inputs, context, metadata, diff_ages)
                            loss = loss + args.lambda_cons * cons_loss

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-loss',     loss.item(),     global_counter[mode])
                writer.add_scalar(f'{mode}/batch-cfm-loss', cfm_loss.item(), global_counter[mode])
                if bae is not None:
                    writer.add_scalar(f'{mode}/batch-bae-loss', bae_loss.item(), global_counter[mode])
                if args.lambda_cons > 0:
                    writer.add_scalar(f'{mode}/batch-cons-loss', cons_loss.item(), global_counter[mode])

                epoch_loss      += loss.item()
                epoch_cfm_loss  += cfm_loss.item()
                epoch_bae_loss  += bae_loss.item()
                epoch_cons_loss += cons_loss.item()
                gpu_alloc    = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                ram_gb       = psutil.Process().memory_info().rss / 1024**3
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss / (step + 1):.4f}",
                    "GPU":  f"{gpu_alloc:.1f}/{gpu_reserved:.1f}GB",
                    "RAM":  f"{ram_gb:.1f}GB",
                })
                global_counter[mode] += 1

            scheduler.step()
            n = len(loader)
            epoch_loss      /= n
            epoch_cfm_loss  /= n
            epoch_bae_loss  /= n
            epoch_cons_loss /= n

            writer.add_scalar(f'{mode}/epoch-loss',     epoch_loss,     epoch)
            writer.add_scalar(f'{mode}/epoch-cfm-loss', epoch_cfm_loss, epoch)
            if bae is not None:
                writer.add_scalar(f'{mode}/epoch-bae-loss', epoch_bae_loss, epoch)
            if args.lambda_cons > 0:
                writer.add_scalar(f'{mode}/epoch-cons-loss', epoch_cons_loss, epoch)

            diff_pred = images_sampling(
                context[0].unsqueeze(0), metadata[0].unsqueeze(0), model,
                diff_ages=diff_ages[0].unsqueeze(0))
            pred_image = torch.clamp(context[0] + diff_pred[0], 0, 1)

            if args.wandb:
                log_dict = {
                    f'{mode}/epoch-loss':   epoch_loss,
                    f'{mode}/epoch-cfm':    epoch_cfm_loss,
                    f'{mode}/predicted_hr': wandb.Image(pred_image[0].detach().cpu()),
                    f'{mode}/img_hr':       wandb.Image(img_hr_2d[0][0].detach().cpu()),
                    f'{mode}/img_lr':       wandb.Image(img_lr_2d[0][0].detach().cpu()),
                }
                if bae is not None:
                    log_dict[f'{mode}/epoch-bae'] = epoch_bae_loss
                if args.lambda_cons > 0:
                    log_dict[f'{mode}/epoch-cons'] = epoch_cons_loss
                wandb.log(log_dict, step=epoch)

        if epoch_loss < min_valid_loss:
            min_valid_loss = epoch_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'tafm2d-best.pth'))

        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, 'tafm2d-last.pth'))
