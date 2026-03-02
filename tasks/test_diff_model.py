import sys
sys.path.append('.')
sys.path.append('./models')

import os
import torch.nn.functional as F
import argparse
import monai
import torch
import pandas as pd
from torch.utils.data import DataLoader
from monai import transforms
from monai.data.image_reader import NumpyReader
from tqdm import tqdm
from models.Diffusion3D import Diffusion3D
from utilities import utils
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two 3D images."""
    mse = ((img1-img2)**2).mean()
    max_pixel = 1.0  # Assuming images are normalized between 0 and 1
    psnr = 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(torch.tensor(mse))
    return psnr.item()

def compute_ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM) for 3D images."""
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    return ssim(img1, img2, data_range=1.0)

def save_nifti(image_tensor, output_path):
    """Save a PyTorch tensor as a NIfTI file."""
    image_numpy = image_tensor.squeeze() * 255.0
    
    nifti_image = nib.Nifti1Image(image_numpy, affine=np.eye(4))
    nib.save(nifti_image, output_path)

def norm_output(output):
    output = (output - output.min()) / (output.max() - output.min())
    return output

def images_sampling(image, metadata, diffusion):
    with torch.no_grad():
        output = diffusion(image=image, metadata=metadata, pred_type="ddim_sample")
        output = torch.clamp(output, -1, 1)
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--cache_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--wandb',       action='store_true')
    parser.add_argument('--run_name',    required=True, type=str)
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="TADM-3D", name = args.run_name)

    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['img_hr', 'img_lr'], reader="NibabelReader"),
        transforms.EnsureChannelFirstD(keys=['img_lr', 'img_hr']),
        transforms.SpacingD(pixdim=1.5, keys=['img_lr', 'img_hr']),
        transforms.ResizeWithPadOrCropD(spatial_size=(128, 128, 128), mode='minimum', keys=['img_lr', 'img_hr'], lazy=True),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['img_hr', 'img_lr']),
    ])

    test_df = pd.read_csv(args.dataset + "test_dataset_updated.csv").to_dict(orient='records')
    testset = monai.data.Dataset(test_df, transforms_fn)
    test_loader = DataLoader(dataset=testset, 
                             num_workers=args.num_workers, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             persistent_workers=True,
                             pin_memory=True)

    diffusion = Diffusion3D().to(DEVICE)
    diffusion.load_state_dict(torch.load(args.diff_ckpt, map_location=DEVICE))
    diffusion.eval()

    mse = 0
    total_psnr = 0
    total_ssim = 0

    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        custom_diffes = []#[24, 48, 72, 96]
        inputs = (batch['img_hr'] - batch['img_lr']).to(DEVICE)
            
        item_name = batch['item_name'][0]
        age = batch['age']
        diff_ages = batch['diff_ages']
        custom_diffes.append(diff_ages)

        if os.path.exists(os.path.join(args.output_dir, f'{item_name}_{int(age[0])}_{int(diff_ages[0])}')):
            continue
        print(item_name)

        for custom_diff in custom_diffes:
            metadata = torch.stack((age, torch.tensor([custom_diff]), batch['patient_condition']), dim=1).float().to(DEVICE)
            context = batch['img_lr'].to(DEVICE)

            diff_pred_image = images_sampling(image=context, metadata=metadata, diffusion=diffusion)
            pred_image = torch.clamp(context + diff_pred_image, 0, 1).cpu().numpy()
            
            mse += ((pred_image - batch['img_hr'].cpu().numpy())**2).mean()
            # Compute metrics
            psnr_value = compute_psnr(pred_image, batch['img_hr'].cpu().numpy())
            ssim_value = compute_ssim(pred_image, batch['img_hr'].cpu().numpy())
            total_psnr += psnr_value
            total_ssim += ssim_value
            
            if not os.path.exists(os.path.join(args.output_dir, f'{item_name}_{int(age[0])}_{int(diff_ages[0])}')):
                os.makedirs(os.path.join(args.output_dir, f'{item_name}_{int(age[0])}_{int(diff_ages[0])}'))

            output_path = os.path.join(args.output_dir, f'{item_name}_{int(age[0])}_{int(diff_ages[0])}/pred_mri_{np.array(custom_diff).astype(int)}.nii')
            save_nifti(pred_image[0], output_path)
        
        save_nifti(batch['img_hr'][0].cpu().numpy(), output_path.replace(output_path.split("/")[-1], 'hr.nii'))
        save_nifti(batch['img_lr'][0].cpu().numpy(), output_path.replace(output_path.split("/")[-1], 'lr.nii'))
        
        print(f"Saved {output_path}")

    print(f"MSE: {mse / (i + 1)}")
    print(f"PSNR: {total_psnr / (i + 1)}")
    print(f"SSIM: {total_ssim / (i + 1)}")

    if args.wandb:
            wandb.log({
                "MSE": mse / (i + 1),
                "PSNR": total_psnr / (i + 1),
                "SSIM": total_ssim / (i + 1)
            })
