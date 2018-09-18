#!/bin/bash
DCASE2018_TASK1_DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task1/TUT-urban-acoustic-scenes-2018-development"
DCASE2018_TASK2_DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task2"

WORKSPACE="/vol/vssp/msos/qk/workspaces/weak_source_separation/dcase2018_task2"

# Create DCASE 2018 Task 2 cross-validation csv
python utils/create_mixture_yaml.py create_dcase2018_task2_cross_validation_csv --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE

# Create mixture yaml file. 
python utils/create_mixture_yaml.py create_mixture_yaml --dcase2018_task1_dataset_dir=$DCASE2018_TASK1_DATASET_DIR --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE

# Create mixed audios
python utils/create_mixed_audio.py --dcase2018_task1_dataset_dir=$DCASE2018_TASK1_DATASET_DIR --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=0

# Calculate features
python utils/features.py logmel --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=0

# Train
MODEL_TYPE="gmp"    # 'gmp' | 'gap' | 'gwrp'
CUDA_VISIBLE_DEVICES=1 python pytorch/main_pytorch.py train --workspace=$WORKSPACE --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --cuda

# Inference
CUDA_VISIBLE_DEVICES=1 python pytorch/main_pytorch.py inference --workspace=$WORKSPACE --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --iteration=10000 --cuda

# Get average stats
python utils/get_avg_stats.py single_fold --workspace=$WORKSPACE --filename=main_pytorch --model_type=gmp --scene_type=dcase2018_task1 --holdout_fold=1 --snr=0

# After calculating stats of all folds, you may run the following command to get averaged stats of all folds. 
# python utils/get_avg_stats.py all_fold --workspace=$WORKSPACE --filename=main_pytorch --model_type=gmp --scene_type=dcase2018_task1 --snr=0


############# Plot figures for paper #############
python utils/plot_for_paper.py waveform --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --cuda
python utils/plot_for_paper.py mel_masks --workspace=$WORKSPACE --model_type=gwrp --scene_type=dcase2018_task1 --snr=0 --holdout_fold=1 --iteration=10000 --cuda
