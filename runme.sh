#!/bin/bash
DCASE2018_TASK1_DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task1/TUT-urban-acoustic-scenes-2018-development"
DCASE2018_TASK2_DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task2"

WORKSPACE="/vol/vssp/msos/qk/workspaces/weak_source_separation/dcase2018_task2"

# Create DCASE 2018 Task 2 cross-validation csv. Only manually verified sound events are used for cross validation. 
python utils/create_mixture_yaml.py create_dcase2018_task2_cross_validation_csv --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE

# Create mixture yaml file of sound events and background noise. 
python utils/create_mixture_yaml.py create_mixture_yaml --dcase2018_task1_dataset_dir=$DCASE2018_TASK1_DATASET_DIR --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE

# Create mixed audios
SNR=0
python utils/create_mixed_audio.py --dcase2018_task1_dataset_dir=$DCASE2018_TASK1_DATASET_DIR --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=$SNR

# Calculate features
python utils/features.py logmel --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=$SNR

# Train
MODEL_TYPE="gmp"    # 'gmp' | 'gap' | 'gwrp'
HOLDOUT_FOLD=1
CUDA_VISIBLE_DEVICES=1 python pytorch/main_pytorch.py train --workspace=$WORKSPACE --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD --cuda

# Inference
CUDA_VISIBLE_DEVICES=1 python pytorch/main_pytorch.py inference --workspace=$WORKSPACE --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --cuda

# Get average statistics
python utils/get_avg_stats.py single_fold --workspace=$WORKSPACE --filename=main_pytorch --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD 

# After train & inference and calculate the statistics of folds 1, 2, 3 and 4, you may run the following command to get averaged statistics of all folds. 
python utils/get_avg_stats.py all_fold --workspace=$WORKSPACE --filename=main_pytorch --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR


############# Plot figures for paper #############
# Visualize waveform & spectrogram
python utils/visualize.py waveform --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD --cuda

# Visualze learned segmentation masks & SED results
CUDA_VISIBLE_DEVICES=1 python utils/visualize.py mel_masks --workspace=$WORKSPACE --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --cuda
