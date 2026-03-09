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
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from monai import transforms
from tqdm import tqdm
from models.FlowMatching3D import FlowMatching3D
from skimage.metrics import structural_similarity as ssim

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_nifti(arr: np.ndarray, path: str):
    arr = arr.squeeze().astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), path)


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    pred = pred.squeeze().astype(np.float64)
    gt   = gt.squeeze().astype(np.float64)
    mse  = float(np.mean((pred - gt) ** 2))
    data_range = gt.max() - gt.min()
    psnr = float(20 * np.log10(data_range / (np.sqrt(mse) + 1e-8))) if mse > 0 else float('inf')
    ssim_val = float(ssim(pred, gt, data_range=data_range))
    return mse, psnr, ssim_val


@torch.no_grad()
def fm_sample(model, context, metadata, diff_ages):
    output = model(image=context, metadata=metadata,
                   pred_type="fm_sample", diff_ages=diff_ages)
    return torch.clamp(output, -1, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     required=True,  type=str)
    parser.add_argument('--fm_ckpt',     required=True,  type=str)
    parser.add_argument('--output_dir',  required=True,  type=str)
    parser.add_argument('--n_samples',   default=5,      type=int)
    parser.add_argument('--split',       default='test', type=str,
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--num_workers', default=4,      type=int)
    parser.add_argument('--interpolant', default='stochastic', type=str,
                        choices=['linear', 'cosine', 'stochastic'])
    parser.add_argument('--solver',      default='heun', type=str,
                        choices=['euler', 'heun'])
    parser.add_argument('--n_steps',     default=20,     type=int)
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
    loader  = DataLoader(dataset, batch_size=1,
                         num_workers=args.num_workers,
                         shuffle=False, pin_memory=False)

    model = FlowMatching3D(
        interpolant_type=args.interpolant,
        n_inference_steps=args.n_steps,
        solver=args.solver,
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.fm_ckpt, map_location=DEVICE))
    model.eval()
    print(f"Loaded FM model from {args.fm_ckpt}")
    print(f"Generating {args.n_samples} samples → {args.output_dir}\n")

    all_mse, all_psnr, all_ssim = [], [], []

    for i, batch in enumerate(tqdm(loader, desc="Sampling")):
        age       = batch['age'].to(DEVICE)
        diff_ages = batch['diff_ages'].to(DEVICE)
        condition = batch['patient_condition'].to(DEVICE)
        context   = batch['img_lr'].to(DEVICE)

        metadata = torch.stack((age, diff_ages, condition), dim=1).float()

        diff_pred = fm_sample(model, context, metadata, diff_ages)
        pred_hr   = context + diff_pred

        pred_np = pred_hr.cpu().numpy()
        gt_np   = batch['img_hr'].cpu().numpy()
        mse, psnr, ssim_val = compute_metrics(pred_np, gt_np)
        all_mse.append(mse);  all_psnr.append(psnr);  all_ssim.append(ssim_val)

        sample_dir = os.path.join(args.output_dir, f'sample_{i:03d}')
        os.makedirs(sample_dir, exist_ok=True)
        save_nifti(batch['img_lr'].cpu().numpy(),  os.path.join(sample_dir, 'lr.nii.gz'))
        save_nifti(gt_np,                          os.path.join(sample_dir, 'hr_gt.nii.gz'))
        save_nifti(pred_np,                        os.path.join(sample_dir, 'hr_pred.nii.gz'))
        save_nifti(diff_pred.cpu().numpy(),        os.path.join(sample_dir, 'diff_pred.nii.gz'))

        print(f"  [{i}] age={float(batch['age'][0]):.1f}y  "
              f"Δ={float(batch['diff_ages'][0]):.2f}y  "
              f"MSE={mse:.4f}  PSNR={psnr:.2f}dB  SSIM={ssim_val:.4f}")

    print("\n" + "="*55)
    print(f"{'Metric':<10}  {'Mean':>10}  {'Std':>10}")
    print("-"*55)
    print(f"{'MSE':<10}  {np.mean(all_mse):>10.4f}  {np.std(all_mse):>10.4f}")
    print(f"{'PSNR(dB)':<10}  {np.mean(all_psnr):>10.2f}  {np.std(all_psnr):>10.2f}")
    print(f"{'SSIM':<10}  {np.mean(all_ssim):>10.4f}  {np.std(all_ssim):>10.4f}")
    print("="*55)

    summary_path = os.path.join(args.output_dir, 'metrics.csv')
    pd.DataFrame({'mse': all_mse, 'psnr': all_psnr, 'ssim': all_ssim}).to_csv(
        summary_path, index_label='sample')
    print(f"\nPer-sample metrics saved to {summary_path}")
