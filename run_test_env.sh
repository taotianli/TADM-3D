#!/bin/bash

echo "=============================="
echo " GPU Info"
echo "=============================="
nvidia-smi

echo ""
echo "=============================="
echo " Installing requirements"
echo "=============================="
cd /nfs/home/ttao/Projects/TADM-3D
pip install -r requirements.txt --ignore-installed torch torchvision

echo ""
echo "=============================="
echo " Python & Package Versions"
echo "=============================="
python -c "
import torch
import monai
import nibabel
import numpy as np
print(f'Python:      OK')
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA avail:  {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'MONAI:       {monai.__version__}')
print(f'NiBabel:     {nibabel.__version__}')
print(f'NumPy:       {np.__version__}')
"

echo ""
echo "=============================="
echo " Environment test complete"
echo "=============================="


