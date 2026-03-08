import sys
sys.path.append('.')
sys.path.append('./models')

import torch

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
import pandas as pd
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from monai import transforms
from tqdm import tqdm
from models.Diffusion3D import Diffusion3D

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_nifti(arr: np.ndarray, path: str):
    """Save array as NIfTI, min-max normalised to [0,1] for easy viewing."""
    arr = arr.squeeze().astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), path)


def ddim_sample(diffusion, context, metadata):
    with torch.no_grad():
        diff = diffusion(image=context, metadata=metadata, pred_type="ddim_sample")
        diff = torch.clamp(diff, -1, 1)
    return diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     required=True,  type=str)
    parser.add_argument('--diff_ckpt',   required=True,  type=str)
    parser.add_argument('--output_dir',  required=True,  type=str)
    parser.add_argument('--n_samples',   default=5,      type=int,
                        help='Number of test samples to generate')
    parser.add_argument('--split',       default='test', type=str,
                        choices=['train', 'valid', 'test'],
                        help='Which split CSV to use')
    parser.add_argument('--num_workers', default=4,      type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['img_hr', 'img_lr'], reader="NibabelReader"),
        transforms.EnsureChannelFirstD(keys=['img_lr', 'img_hr']),
        transforms.SpacingD(pixdim=1.5, keys=['img_lr', 'img_hr']),
        transforms.ResizeWithPadOrCropD(spatial_size=(128, 128, 128),
                                        mode='minimum',
                                        keys=['img_lr', 'img_hr'],
                                        lazy=True),
        transforms.NormalizeIntensityD(keys=['img_hr', 'img_lr'],
                                       nonzero=True, channel_wise=True),
    ])

    csv_path = os.path.join(args.dataset, f'{args.split}_dataset.csv')
    df = pd.read_csv(csv_path)
    for col in ('img_hr', 'img_lr'):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda p: p if os.path.isabs(p) else os.path.join(args.dataset, p)
            )
    records = df.head(args.n_samples).to_dict(orient='records')

    dataset = monai.data.Dataset(records, transforms_fn)
    loader  = DataLoader(dataset,
                         batch_size=1,
                         num_workers=args.num_workers,
                         shuffle=False,
                         pin_memory=False)

    diffusion = Diffusion3D().to(DEVICE)
    diffusion.load_state_dict(torch.load(args.diff_ckpt, map_location=DEVICE))
    diffusion.eval()
    print(f"Loaded diffusion model from {args.diff_ckpt}")
    print(f"Generating {args.n_samples} samples → {args.output_dir}\n")

    for i, batch in enumerate(tqdm(loader, desc="Sampling")):
        age         = batch['age'].to(DEVICE)
        diff_ages   = (batch['diff_ages'] / 12.0).to(DEVICE)   # months → years
        condition   = batch['patient_condition'].to(DEVICE)
        context     = batch['img_lr'].to(DEVICE)                # LR image as condition

        metadata = torch.stack((age, diff_ages, condition), dim=1).float()

        diff_pred = ddim_sample(diffusion, context, metadata)
        pred_hr   = context + diff_pred                         # LR + predicted diff

        sample_dir = os.path.join(args.output_dir, f'sample_{i:03d}')
        os.makedirs(sample_dir, exist_ok=True)

        save_nifti(batch['img_lr'].cpu().numpy(),  os.path.join(sample_dir, 'lr.nii.gz'))
        save_nifti(batch['img_hr'].cpu().numpy(),  os.path.join(sample_dir, 'hr_gt.nii.gz'))
        save_nifti(pred_hr.cpu().numpy(),          os.path.join(sample_dir, 'hr_pred.nii.gz'))
        save_nifti(diff_pred.cpu().numpy(),        os.path.join(sample_dir, 'diff_pred.nii.gz'))

        age_val  = float(batch['age'][0])
        diff_val = float(batch['diff_ages'][0])
        print(f"  [{i}] age={age_val:.1f}y  diff={diff_val:.1f}mo  → {sample_dir}")

    print("\nDone.")
