"""
Training script for TAFM-3D (Flow Matching version of TADM-3D).

Key differences from train_diff_model.py:
  - CFM loss: MSE between predicted and target velocity
  - Adaptive time sampler annealing each epoch
  - Innovation 5: Consistency Regularization Loss
  - diff_ages passed to model for temporal-aware OT path
"""

import sys
sys.path.append('.')
sys.path.append('./models')

import torch

# Workaround for PyTorch 2.10 bug: PrecompileCacheArtifact gets registered twice
# when torch._dynamo is lazily imported after monai/generative already triggered it.
import torch.compiler._cache as _tc
@classmethod  # type: ignore[misc]
def _safe_register(cls, artifact_cls):
    if artifact_cls.type() not in cls._artifact_types:
        cls._artifact_types[artifact_cls.type()] = artifact_cls
    return artifact_cls
_tc.CacheArtifactFactory.register = _safe_register
del _tc, _safe_register

import os
import argparse
import random
import numpy as np
import psutil

import monai
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from tqdm import tqdm

from models.FlowMatching3D import FlowMatching3D
from models.BAE3D import BAE3D
from utilities.lr_scheduler import LinearWarmupCosineAnnealingLR
from utilities import utils, const

import wandb

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def norm_output(output):
    return (output - output.min()) / (output.max() - output.min() + 1e-8)


@torch.no_grad()
def images_sampling(image, metadata, model, diff_ages=None):
    """Generate a predicted residual via ODE integration."""
    output = model(image=image, metadata=metadata,
                   pred_type="fm_sample", diff_ages=diff_ages)
    return torch.clamp(output, -1, 1)


# ---------------------------------------------------------------------------
# Innovation 5 — Consistency Regularization Loss
# ---------------------------------------------------------------------------

def consistency_loss(model, x1, image, metadata, diff_ages):
    """
    Consistency Regularization: two independently sampled (x_t, t) pairs
    from the same x_1 should produce velocity predictions consistent with
    the same underlying flow.

        x1_hat_a = x_t_a + (1 - t_a) * v_theta(x_t_a, t_a)   [with grad]
        x1_hat_b = x_t_b + (1 - t_b) * v_theta(x_t_b, t_b)   [no grad, target]
        L_cons = MSE(x1_hat_a, x1_hat_b.detach())

    The second pair runs under torch.no_grad() since it is used only as a
    detached target, halving the extra memory overhead vs. a naive two-pass.
    """
    embeddings = model.embed_model(image)

    # first pass — gradients flow through this one
    x_t_a, t_a, _ = model(x=x1, pred_type="get_train_sample", diff_ages=diff_ages)
    v_pred_a = model.model(x_t_a, t=t_a, image=image,
                           embeddings=embeddings, metadata=metadata)
    x1_hat_a = x_t_a + (1.0 - t_a.view(-1, 1, 1, 1, 1)) * v_pred_a

    # second pass — detached target, no gradient needed
    with torch.no_grad():
        x_t_b, t_b, _ = model(x=x1, pred_type="get_train_sample", diff_ages=diff_ages)
        v_pred_b = model.model(x_t_b, t=t_b, image=image,
                               embeddings=embeddings, metadata=metadata)
        x1_hat_b = x_t_b + (1.0 - t_b.view(-1, 1, 1, 1, 1)) * v_pred_b

    return F.mse_loss(x1_hat_a, x1_hat_b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     required=True,  type=str)
    parser.add_argument('--cache_dir',   required=True,  type=str)
    parser.add_argument('--output_dir',  required=True,  type=str)
    parser.add_argument('--fm_ckpt',     default=None,   type=str)
    parser.add_argument('--bae_ckpt',    default=None,   type=str)
    parser.add_argument('--num_workers', default=8,      type=int)
    parser.add_argument('--n_epochs',    default=500,    type=int)
    parser.add_argument('--batch_size',  default=16,     type=int)
    parser.add_argument('--lr',          default=1e-4,   type=float)
    parser.add_argument('--wandb',       action='store_true')
    parser.add_argument('--run_name',    required=True,  type=str)
    parser.add_argument('--lambda_cons', default=0.1,    type=float,
                        help='Weight for consistency regularization loss')
    parser.add_argument('--interpolant', default='stochastic', type=str,
                        choices=['linear', 'cosine', 'stochastic'])
    parser.add_argument('--sigma_min',         default=0.01,  type=float)
    parser.add_argument('--no_time_annealing', action='store_true',
                        help='Disable adaptive time sampler annealing (ablation)')
    parser.add_argument('--no_ot_scaling',     action='store_true',
                        help='Disable temporal-aware OT source scaling (ablation)')
    parser.add_argument('--no_tpg',            action='store_true',
                        help='Disable Temporal Progression Gate (ablation)')
    parser.add_argument('--no_cross_attn',     action='store_true',
                        help='Disable Cross-Attention Gate at bottleneck (ablation)')
    parser.add_argument('--solver',      default='heun', type=str,
                        choices=['euler', 'heun'])
    parser.add_argument('--n_steps',     default=20,     type=int,
                        help='ODE steps at inference')
    parser.add_argument('--sigma_min',   default=0.01,   type=float,
                        help='Sigma_min for stochastic interpolant')
    # Ablation flags — disable individual innovations
    parser.add_argument('--no_tpg',           action='store_true',
                        help='Disable Temporal Progression Gate (Innovation 4)')
    parser.add_argument('--no_cross_attn',    action='store_true',
                        help='Disable Cross-Attention Gate at bottleneck (Innovation 3)')
    parser.add_argument('--no_ot_scaling',    action='store_true',
                        help='Disable temporal-aware OT path scaling (Innovation 2)')
    parser.add_argument('--no_time_annealing', action='store_true',
                        help='Disable adaptive time sampler annealing (Innovation 1)')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="TAFM-3D", name=args.run_name)

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
                              persistent_workers=args.num_workers > 0,
                              pin_memory=False)
    valid_loader = DataLoader(validset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False,
                              persistent_workers=args.num_workers > 0,
                              pin_memory=False)

    model = FlowMatching3D(
        interpolant_type=args.interpolant,
        sigma_min=args.sigma_min,
        n_inference_steps=args.n_steps,
        solver=args.solver,
        use_tpg=not args.no_tpg,
        use_cross_attn=not args.no_cross_attn,
        use_ot_scaling=not args.no_ot_scaling,
    ).to(DEVICE)

    if args.fm_ckpt is not None:
        model.load_state_dict(torch.load(args.fm_ckpt, map_location=DEVICE))
        print(f"Loaded checkpoint from {args.fm_ckpt}")

    bae = None
    if args.bae_ckpt is not None:
        bae = BAE3D().to(DEVICE)
        bae.load_state_dict(torch.load(args.bae_ckpt, map_location=DEVICE))
        bae.eval()
        for p in bae.parameters():
            p.requires_grad = False
        print(f"Loaded and froze BAE from {args.bae_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=500)
    scaler    = GradScaler('cuda')
    writer    = SummaryWriter()

    global_counter = {'train': 0, 'valid': 0}
    loaders        = {'train': train_loader, 'valid': valid_loader}
    min_valid_loss = np.inf

    for epoch in range(args.n_epochs):

        # Anneal adaptive time sampler (Innovation 1) — skip if disabled
        if not args.no_time_annealing:
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

                    if not reverse:
                        inputs       = (batch['img_hr'] - batch['img_lr']).to(DEVICE)
                        diff_ages_yr = batch['diff_ages']
                        metadata     = torch.stack(
                            (batch['age'], diff_ages_yr, batch['patient_condition']), dim=1
                        ).float().to(DEVICE)
                        context   = batch['img_lr'].to(DEVICE)
                        diff_ages = diff_ages_yr.to(DEVICE)
                    else:
                        inputs       = (batch['img_lr'] - batch['img_hr']).to(DEVICE)
                        diff_ages_yr = batch['diff_ages']
                        metadata     = torch.stack(
                            (batch['age'] + diff_ages_yr,
                             -diff_ages_yr,
                             batch['patient_condition']), dim=1
                        ).float().to(DEVICE)
                        context   = batch['img_hr'].to(DEVICE)
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
                            t_ = t.view(-1, 1, 1, 1, 1)
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
                            gt_diff_age = batch['diff_ages'].to(DEVICE).unsqueeze(1).float()
                            bae_loss = F.mse_loss(predicted_diff_age.float(), gt_diff_age)
                            loss = loss + bae_loss

                        cons_loss = torch.tensor(0.0, device=DEVICE)
                        if args.lambda_cons > 0 and mode == 'train':
                            cons_loss = consistency_loss(
                                model, inputs, context, metadata, diff_ages=diff_ages)
                            loss = loss + args.lambda_cons * cons_loss

                if mode == 'train':
                    scaler.scale(loss).backward()
                    # unscale before clipping so the threshold is in real gradient space
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                # skip NaN batches silently (weights already protected by clipping,
                # but NaN can still appear from extreme inputs)
                loss_val     = loss.item()
                cfm_loss_val = cfm_loss.item()
                if not (loss_val == loss_val):   # NaN check
                    global_counter[mode] += 1
                    continue

                writer.add_scalar(f'{mode}/batch-loss',      loss_val,        global_counter[mode])
                writer.add_scalar(f'{mode}/batch-cfm-loss',  cfm_loss_val,    global_counter[mode])
                if bae is not None:
                    writer.add_scalar(f'{mode}/batch-bae-loss',  bae_loss.item(),  global_counter[mode])
                if args.lambda_cons > 0:
                    writer.add_scalar(f'{mode}/batch-cons-loss', cons_loss.item(), global_counter[mode])

                epoch_loss      += loss_val
                epoch_cfm_loss  += cfm_loss_val
                epoch_bae_loss  += bae_loss.item()
                epoch_cons_loss += cons_loss.item()

                gpu_alloc   = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                ram_gb      = psutil.Process().memory_info().rss / 1024**3
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss / (step + 1):.4f}",
                    "GPU":  f"{gpu_alloc:.1f}/{gpu_reserved:.1f}GB",
                    "RAM":  f"{ram_gb:.1f}GB",
                })
                global_counter[mode] += 1

            n = len(loader)
            epoch_loss      /= n
            epoch_cfm_loss  /= n
            epoch_bae_loss  /= n
            epoch_cons_loss /= n

            writer.add_scalar(f'{mode}/epoch-loss',     epoch_loss,     epoch)
            writer.add_scalar(f'{mode}/epoch-cfm-loss', epoch_cfm_loss, epoch)
            if bae is not None:
                writer.add_scalar(f'{mode}/epoch-bae-loss',  epoch_bae_loss,  epoch)
            if args.lambda_cons > 0:
                writer.add_scalar(f'{mode}/epoch-cons-loss', epoch_cons_loss, epoch)

            diff_pred_image = images_sampling(
                context[0].unsqueeze(0), metadata[0].unsqueeze(0), model,
                diff_ages=diff_ages[0].unsqueeze(0))
            pred_image = context[0] + diff_pred_image[0]

            if args.wandb:
                log_dict = {
                    f'{mode}/epoch-loss': epoch_loss,
                    f'{mode}/epoch-cfm':  epoch_cfm_loss,
                    f'{mode}/predicted_hr': wandb.Image(
                        pred_image[0][:, :, pred_image.shape[2] // 2].detach().cpu()),
                    f'{mode}/img_hr': wandb.Image(
                        batch['img_hr'][0][0][:, :, batch['img_hr'].shape[3] // 2].detach().cpu()),
                    f'{mode}/img_lr': wandb.Image(
                        batch['img_lr'][0][0][:, :, batch['img_lr'].shape[3] // 2].detach().cpu()),
                }
                if bae is not None:
                    log_dict[f'{mode}/epoch-bae'] = epoch_bae_loss
                if args.lambda_cons > 0:
                    log_dict[f'{mode}/epoch-cons'] = epoch_cons_loss
                wandb.log(log_dict, step=epoch)

        # step once per epoch (outside the mode loop)
        scheduler.step()

        if epoch_loss < min_valid_loss:
            min_valid_loss = epoch_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'tafm-best.pth'))

        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, 'tafm-last.pth'))
