"""
Inference / evaluation script for FlowMatching2D (2D version of TAFM-3D).
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
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import wandb

from models.FlowMatching2D import FlowMatching2D

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


def save_png(image_array, output_path):
    """Save a 2D numpy array as a PNG (values in [0,1])."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imsave(output_path, image_array.squeeze(), cmap='gray', vmin=0, vmax=1)


def extract_slice(vol, axis):
    """Extract centre slice (B, C, D, H, W) → (B, C, H, W)."""
    mid = vol.shape[axis + 2] // 2
    if axis == 0:
        return vol[:, :, mid, :, :]
    elif axis == 1:
        return vol[:, :, :, mid, :]
    else:
        return vol[:, :, :, :, mid]


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
    parser.add_argument('--slice_axis',  default=2,     type=int,
                        help='Axis for 2D slice extraction (0=sagittal,1=coronal,2=axial)')
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

    test_df     = pd.read_csv(args.dataset + "test_dataset_updated.csv").to_dict(orient='records')
    testset     = monai.data.Dataset(test_df, transforms_fn)
    test_loader = DataLoader(testset, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=False,
                             persistent_workers=args.num_workers > 0, pin_memory=False)

    model = FlowMatching2D(n_inference_steps=args.n_steps, solver=args.solver).to(DEVICE)
    model.load_state_dict(torch.load(args.fm_ckpt, map_location=DEVICE))
    model.eval()

    total_mse = total_psnr = total_ssim = 0.0

    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        item_name = batch['item_name'][0]
        age       = batch['age']
        diff_ages = batch['diff_ages'] / 12.0

        out_dir = os.path.join(args.output_dir,
                               f'{item_name}_{int(age[0])}_{int(diff_ages[0])}')
        if os.path.exists(out_dir):
            continue

        # Extract 2D slices
        img_hr_2d = extract_slice(batch['img_hr'], args.slice_axis)
        img_lr_2d = extract_slice(batch['img_lr'], args.slice_axis)

        metadata      = torch.stack(
            (age, diff_ages, batch['patient_condition']), dim=1
        ).float().to(DEVICE)
        context       = img_lr_2d.to(DEVICE)
        diff_ages_dev = diff_ages.to(DEVICE)

        diff_pred  = images_sampling(context, metadata, model, diff_ages_dev)
        pred_image = torch.clamp(context + diff_pred, 0, 1).cpu().numpy()
        gt_image   = img_hr_2d.numpy()

        total_mse  += ((pred_image - gt_image) ** 2).mean()
        total_psnr += compute_psnr(pred_image, gt_image)
        total_ssim += compute_ssim(pred_image, gt_image)

        os.makedirs(out_dir, exist_ok=True)
        save_png(pred_image[0],
                 os.path.join(out_dir, f'pred_mri_{int(diff_ages[0])}.png'))
        save_png(gt_image[0],
                 os.path.join(out_dir, 'hr.png'))
        save_png(img_lr_2d.numpy()[0],
                 os.path.join(out_dir, 'lr.png'))

    n = i + 1
    print(f"MSE:  {total_mse  / n:.6f}")
    print(f"PSNR: {total_psnr / n:.4f} dB")
    print(f"SSIM: {total_ssim / n:.4f}")

    if args.wandb:
        wandb.log({"MSE":  total_mse  / n,
                   "PSNR": total_psnr / n,
                   "SSIM": total_ssim / n})
