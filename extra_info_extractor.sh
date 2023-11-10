#!/bin/bash

python train.py --config configs/nerf_synthetic.yaml path=/home/operel/Code/deploy/data/nerf_synthetic/ficus
python train.py --config configs/nerf_synthetic.yaml path=/home/operel/Code/deploy/data/nerf_synthetic/materials
python train.py --config configs/nerf_synthetic.yaml path=/home/operel/Code/deploy/data/nerf_synthetic/ship
python train.py --config configs/nerf_synthetic.yaml path=/home/operel/Code/deploy/data/nerf_synthetic/lego
python train.py --config configs/nerf_synthetic.yaml path=/home/operel/Code/deploy/data/nerf_synthetic/drums
python train.py --config configs/nerf_synthetic.yaml path=/home/operel/Code/deploy/data/nerf_synthetic/chair
python train.py --config configs/nerf_synthetic.yaml path=/home/operel/Code/deploy/data/nerf_synthetic/hotdog
python train.py --config configs/nerf_synthetic.yaml path=/home/operel/Code/deploy/data/nerf_synthetic/mic

python train.py --config configs/base.yaml path=/home/operel/Code/deploy/data/truck
python train.py --config configs/colmap.yaml path=/home/operel/Code/deploy/data/bicycle initialization.method=colmap
