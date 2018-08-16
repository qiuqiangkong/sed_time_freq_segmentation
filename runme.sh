DCASE2018_TASK1_DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task1/TUT-urban-acoustic-scenes-2018-development"
DCASE2018_TASK2_DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task2"

WORKSPACE="/vol/vssp/msos/qk/workspaces/weak_source_separation/dcase2018_task2"

# Create events cv fold
python create_mixture_yaml.py create_dcase2018_cv_folds --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE

# Create mixture cv file
python create_mixture_yaml.py create_mixed_audio --dcase2018_task1_dataset_dir=$DCASE2018_TASK1_DATASET_DIR --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE

# Create mixed audios
python create_mixed_audio.py --dcase2018_task1_dataset_dir=$DCASE2018_TASK1_DATASET_DIR --dcase2018_task2_dataset_dir=$DCASE2018_TASK2_DATASET_DIR --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=20

# Calculate features
python features.py logmel --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=20

# Train
CUDA_VISIBLE_DEVICES=1 python tmp01.py train --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=20 --holdout_fold=1 --cuda

CUDA_VISIBLE_DEVICES=1 python tmp01.py inference --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=20 --holdout_fold=1 --iteration=200 --cuda

# Get average stats
python get_avg_stats.py --workspace=$WORKSPACE --filename=tmp02b --scene_type=dcase2018_task1 --snr=20