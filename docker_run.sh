#source /opt/conda/etc/profile.d/conda.sh
#conda activate watermarknn
#python train.py --batch_size 100 --max_epochs 5 --runname train --wm_batch_size 2 --wmtrain --test_db_path /data/output_dataset --save_model filtered_anomaly_model.t7 --save_dir /data
#python predict.py --model_path /data/filtered_anomaly_model.t7 --save_dir /data
#python fine-tune.py --lr 0.01 --load_path /data/filtered_anomaly_model.t7 --save_dir /data --save_model ftll.t7 --runname fine.tune.last.layer
#python fine-tune.py --lr 0.01 --load_path /data/filtered_anomaly_model.t7 --save_dir /data --save_model ftal.t7 --runname fine.tune.all.layers --tunealllayers
#python fine-tune.py --lr 0.01 --load_path /data/filtered_anomaly_model.t7 --save_dir /data --save_model rtll.t7 --runname reinit.last.layer --reinitll
#python fine-tune.py --lr 0.01 --load_path /data/filtered_anomaly_model.t7 --save_dir /data --save_model rtal.t7 --runname reinit_all.layers --reinitll --tunealllayers

#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate watermarknn

# Set CUDA environment variables for better performance
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDNN_BENCHMARK=1       # Enable cuDNN benchmarking
export CUDA_LAUNCH_BLOCKING=0  # Disable launch blocking for parallel execution
export TORCH_CUDNN_V8_API_ENABLED=1  # Enable cuDNN v8 API

# Training with watermarks
echo "Starting training with watermarks..."
python train.py \
    --batch_size 128 \
    --max_epochs 5 \
    --runname train \
    --wm_batch_size 4 \
    --wmtrain \
    --test_db_path /data/output_dataset \
    --save_model filtered_anomaly_model.t7 \
    --save_dir /data \
    --lr 0.05

# Testing the model
echo "Testing trained model..."
python predict.py \
    --model_path /data/filtered_anomaly_model.t7 \
    --save_dir /data

# Fine-tuning variants
echo "Performing FTLL (Fine-Tune Last Layer)..."
python fine-tune.py \
    --lr 0.01 \
    --load_path /data/filtered_anomaly_model.t7 \
    --save_dir /data \
    --save_model ftll.t7 \
    --runname fine.tune.last.layer

echo "Performing FTAL (Fine-Tune All Layers)..."
python fine-tune.py \
    --lr 0.01 \
    --load_path /data/filtered_anomaly_model.t7 \
    --save_dir /data \
    --save_model ftal.t7 \
    --runname fine.tune.all.layers \
    --tunealllayers

echo "Performing RTLL (Retrain Last Layer)..."
python fine-tune.py \
    --lr 0.01 \
    --load_path /data/filtered_anomaly_model.t7 \
    --save_dir /data \
    --save_model rtll.t7 \
    --runname reinit.last.layer \
    --reinitll

echo "Performing RTAL (Retrain All Layers)..."
python fine-tune.py \
    --lr 0.01 \
    --load_path /data/filtered_anomaly_model.t7 \
    --save_dir /data \
    --save_model rtal.t7 \
    --runname reinit_all.layers \
    --reinitll \
    --tunealllayers

echo "All tasks completed successfully!"