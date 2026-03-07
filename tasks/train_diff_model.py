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
import monai
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from monai.data.image_reader import NumpyReader
from tqdm import tqdm
from models.Diffusion3D import Diffusion3D
from models.BAE3D import BAE3D
from utilities.lr_scheduler import LinearWarmupCosineAnnealingLR

from utilities import utils
from utilities import const

import wandb
import numpy as np
import random
import psutil

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def norm_output(output):
    output = (output - output.min())/(output.max() - output.min())
    return output

def images_sampling(image, metadata, diffusion):
    with torch.no_grad():
        output = diffusion(image=image, metadata=metadata, pred_type="ddim_sample")
        output = torch.clamp(output, -1, 1)
        return output

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  required=True, type=str)
    parser.add_argument('--cache_dir',  required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--diff_ckpt',   default=None, type=str)
    parser.add_argument('--bae_ckpt',    default=None, type=str)
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=500,     type=int)
    parser.add_argument('--batch_size',  default=16,    type=int)
    parser.add_argument('--lr',          default=1e-4,  type=float)
    parser.add_argument('--wandb',       action='store_true')
    parser.add_argument('--run_name',    required=True, type=str)
    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project="TADM-3D", name = args.run_name)

    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['img_hr', 'img_lr'], reader="NibabelReader"),
        transforms.EnsureChannelFirstD(keys=['img_lr', 'img_hr']), 
        transforms.SpacingD(pixdim=1.5, keys=[ 'img_lr', 'img_hr']),
        transforms.ResizeWithPadOrCropD(spatial_size=(128, 128, 128), mode='minimum', keys=[ 'img_lr', 'img_hr'], lazy=True),
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

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=args.num_workers > 0,
                              pin_memory=False)
    
    valid_loader = DataLoader(dataset=validset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              persistent_workers=args.num_workers > 0, 
                              pin_memory=False)
    
    diffusion = Diffusion3D().to(DEVICE)

    if args.diff_ckpt is not None:
        diffusion.load_state_dict(torch.load(args.diff_ckpt, map_location=DEVICE))
        print(f"Resumed diffusion model from {args.diff_ckpt}")

    # Load and freeze BAE model if checkpoint is provided
    bae = None
    if args.bae_ckpt is not None:
        bae = BAE3D().to(DEVICE)
        bae.load_state_dict(torch.load(args.bae_ckpt, map_location=DEVICE))
        bae.eval()
        for param in bae.parameters():
            param.requires_grad = False
        print(f"Loaded and froze BAE model from {args.bae_ckpt}")

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                warmup_epochs=50,
                                                max_epochs=500)
    
    scaler = GradScaler('cuda')
    
    writer = SummaryWriter()
    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }

    min_valid_loss = np.inf

    for epoch in range(args.n_epochs):
        
        for mode in loaders.keys():
            
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()

            epoch_loss = 0
            epoch_diff_loss = 0
            epoch_bae_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in progress_bar:
                            
                with autocast('cuda', enabled=True):       
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)

                    reverse = True if (random.randint(0, 1) == 1 and mode == 'train') else False
                    
                    inputs = (batch['img_hr']-batch['img_lr']).to(DEVICE) if not reverse else (batch['img_lr']-batch['img_hr']).to(DEVICE)

                    diff_ages_yr = batch['diff_ages'] / 12.0
                    metadata = torch.stack((batch['age'], diff_ages_yr, batch['patient_condition']), dim=1).float().to(DEVICE) if not reverse else torch.stack((batch['age']+diff_ages_yr, -diff_ages_yr, batch['patient_condition']), dim=1).float().to(DEVICE)
                    context = batch['img_lr'].to(DEVICE) if not reverse else batch['img_hr'].to(DEVICE)
                                        
                    with torch.set_grad_enabled(mode == 'train'):
                        x_t, t, noise = diffusion(x=inputs, pred_type="q_sample")
                        pred_inputs = diffusion(x=x_t, step=t, image=context, metadata=metadata, pred_type="denoise")

                        mse_diff_loss = F.mse_loss( pred_inputs.float(), inputs.float() )
                        loss = mse_diff_loss

                        if bae is not None:
                            if not reverse:
                                pred_hr = context + pred_inputs
                                bae_age = batch['age'].to(DEVICE)
                                predicted_diff_age = bae(context, pred_hr, bae_age, batch['patient_condition'].to(DEVICE))
                            else:
                                pred_lr = context - pred_inputs
                                bae_age = batch['age'].to(DEVICE)
                                predicted_diff_age = bae(pred_lr, context, bae_age, batch['patient_condition'].to(DEVICE))

                            gt_diff_age = (batch['diff_ages'] / 12.0).to(DEVICE).unsqueeze(1).float()
                            bae_loss = F.mse_loss(predicted_diff_age.float(), gt_diff_age)
                            loss = mse_diff_loss + bae_loss
                        else:
                            bae_loss = torch.tensor(0.0)

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                writer.add_scalar(f'{mode}/batch-mse-diff', mse_diff_loss.item(), global_counter[mode])
                if bae is not None:
                    writer.add_scalar(f'{mode}/batch-mse-bae', bae_loss.item(), global_counter[mode])

                epoch_loss += loss.item()
                epoch_diff_loss += mse_diff_loss.item()
                epoch_bae_loss += bae_loss.item() if bae is not None else 0.0

                gpu_alloc = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                ram_gb = psutil.Process().memory_info().rss / 1024**3
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss / (step + 1):.4f}",
                    "GPU": f"{gpu_alloc:.1f}/{gpu_reserved:.1f}GB",
                    "RAM": f"{ram_gb:.1f}GB",
                })
                global_counter[mode] += 1

            scheduler.step()
            epoch_loss = epoch_loss / len(loader)
            epoch_diff_loss = epoch_diff_loss / len(loader)
            epoch_bae_loss = epoch_bae_loss / len(loader)

            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)
            writer.add_scalar(f'{mode}/epoch-mse-diff', epoch_diff_loss, epoch)
            if bae is not None:
                writer.add_scalar(f'{mode}/epoch-mse-bae', epoch_bae_loss, epoch)

            diff_pred_image = images_sampling(image=context[0].unsqueeze(0), metadata=metadata[0].unsqueeze(0), diffusion=diffusion)
            pred_image = context[0] + diff_pred_image[0]

            img_hr = batch['img_hr'][0]
            img_lr = batch['img_lr'][0]
            
            if args.wandb:
                log_dict = {
                    f'{mode}/epoch-mse': epoch_loss,
                    f'{mode}/epoch-mse-diff': epoch_diff_loss,
                    f'{mode}/predicted_hr': wandb.Image(pred_image[0][:,:,pred_image.shape[3]//2].detach().cpu()),
                    f'{mode}/img_hr': wandb.Image(img_hr[0][:,:,img_hr.shape[3]//2].detach().cpu()),
                    f'{mode}/img_lr': wandb.Image(img_lr[0][:,:,img_lr.shape[3]//2].detach().cpu()),
                    f'{mode}/predicted_diff': wandb.Image(diff_pred_image[0][0][:,:,diff_pred_image.shape[3]//2].detach().cpu()),
                    f'{mode}/diff': wandb.Image(inputs[0][0][:,:,diff_pred_image.shape[3]//2].detach().cpu())
                }
                if bae is not None:
                    log_dict[f'{mode}/epoch-mse-bae'] = epoch_bae_loss
                wandb.log(log_dict, step=epoch)

        if epoch_loss < min_valid_loss:
            min_valid_loss = epoch_loss
            savepath = os.path.join(args.output_dir, f'unet-best.pth')
            torch.save(diffusion.state_dict(), savepath)
        
        savepath = os.path.join(args.output_dir, f'unet-last.pth')
        torch.save(diffusion.state_dict(), savepath)
