#!/bin/bash


module load python/3.7
#python run_cnn.py
python run_hyper.py

# sbatch --partition=gpu --cpus-per-task=8 --gres=gpu:k80:2  --mem=40g run_cnn_sbacth.sh --time 10:00:00
