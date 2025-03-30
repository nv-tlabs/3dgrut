#!/bin/bash

# Directly Print 

#  - Training Time: 
echo "- Train (s)" && grep "Training Statistics" -A 5 train_*.log | awk 'NR % 7 == 5' | awk -F' ' '{print $6}'

#  - Testing PSNR: 
echo "- Test PSNR" && grep "Test Metrics"        -A 5 train_*.log | awk 'NR % 7 == 5' | awk -F' ' '{print $2}'

#  - Testing SSIM: 
echo "- Test SSIM" && grep "Test Metrics"        -A 5 train_*.log | awk 'NR % 7 == 5' | awk -F' ' '{print $4}'

#  - Rendering Frame Time: 
echo "- Frame Time (ms)" && grep "Test Metrics"  -A 5 render_*.log | awk 'NR % 7 == 5' | awk -F' ' '{print $10}'
