"""
Inference / evaluation script for TAFM-3D.
"""

import sys
sys.path.append('.')
sys.path.append('./models')

import os
import argparse
import numpy as np

import monai
import torch
import pandas as pd
from torch.utils.data import DataLoader
from monai import transforms
from monai.data.image_reader import NumpyReader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import nibabel as nib
import wandb

from models.FlowMatching3D import FlowMatching3D

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0) - 10 * np.log10(mse)


def compute_ssim(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    return ssim(img1, img2, data_range=1.0)


def save_nifti(image_tensor, output_path):
    image_numpy = image_tensor.squeeze() * 255.0
    nib.save(nib.Nifti1Image(image_numpy, affine=np.eye(4)), output_path)


@torch.no_grad()
def images_sampling(image, metadata, model, diff_ages=None):
    output = model(image=image, metadata=metadata,
                   pred_type="fm_sample", diff_ages=diff_ages)
    return torch.clamp(output, -1, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     required=True, type=str)
    parser.add_argument('--cache_dir',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--fm_ckpt',     required=True, type=str)
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--batch_size',  default=1,     type=int)
    parser.add_argument('--wandb',       action='store_true')
    parser.add_argument('--run_name',    required=True, type=str)
    parser.add_argument('--n_steps',     default=20,    type=int)
    parser.add_argument('--solver',      default='heun', type=str,
                        choices=['euler', 'heun'])
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="TAFM-3D", name=args.run_name)

    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['img_hr', 'img_lr'], reader="NibabelReader"),
        transforms.EnsureChannelFirstD(keys=['img_lr', 'img_hr']),
        transforms.SpacingD(pixdim=1.5, keys=['img_lr', 'img_hr']),
        transforms.ResizeWithPadOrCropD(spatial_size=(128, 128, 128),
                                        mode='minimum', keys=['img_lr', 'img_hr'], lazy=True),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['img_hr', 'img_lr']),
    ])

    test_df  = pd.read_csv(args.dataset + "test_dataset_updated.csv").to_dict(orient='records')
    testset  = monai.data.Dataset(test_df, transforms_fn)
    test_loader = DataLoader(testset, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=False,
                             persistent_workers=True, pin_memory=True)

    model = FlowMatching3D(n_inference_steps=args.n_steps, solver=args.solver).to(DEVICE)
    model.load_state_dict(torch.load(args.fm_ckpt, map_location=DEVICE))
    model.eval()

    total_mse = total_psnr = total_ssim = 0.0

    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        item_name = batch['item_name'][0]
        age       = batch['age']
        diff_ages = batch['diff_ages']

        out_dir = os.path.join(args.output_dir,
                               f'{item_name}_{int(age[0])}_{int(diff_ages[0])}')
        if os.path.exists(out_dir):
            continue

        metadata  = torch.stack(
            (age, diff_ages, batch['patient_condition']), dim=1
        ).float().to(DEVICE)
        context   = batch['img_lr'].to(DEVICE)
        diff_ages_dev = diff_ages.to(DEVICE)

        diff_pred = images_sampling(context, metadata, model, diff_ages_dev)
        pred_image = torch.clamp(context + diff_pred, 0, 1).cpu().numpy()
        gt_image   = batch['img_hr'].cpu().numpy()

        total_mse  += ((pred_image - gt_image) ** 2).mean()
        total_psnr += compute_psnr(pred_image, gt_image)
        total_ssim += compute_ssim(pred_image, gt_image)

        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir,
                                   f'pred_mri_{int(diff_ages[0])}.nii')
        save_nifti(pred_image[0], output_path)
        save_nifti(batch['img_hr'][0].cpu().numpy(),
                   os.path.join(out_dir, 'hr.nii'))
        save_nifti(batch['img_lr'][0].cpu().numpy(),
                   os.path.join(out_dir, 'lr.nii'))

    n = i + 1
    print(f"MSE:  {total_mse  / n:.6f}")
    print(f"PSNR: {total_psnr / n:.4f} dB")
    print(f"SSIM: {total_ssim / n:.4f}")

    if args.wandb:
        wandb.log({"MSE": total_mse / n,
                   "PSNR": total_psnr / n,
                   "SSIM": total_ssim / n})
