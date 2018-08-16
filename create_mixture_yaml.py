import os
import sys
import numpy as np
import pandas as pd
import argparse
import random
import yaml
from collections import OrderedDict
import pandas as pd

from utilities import create_folder, read_audio
import config


def create_validation_folds(args):
    """Create validation file with folds and write out to validate_meta.csv
    """

    # Arguments & parameters
    dcase2018_task2_dataset_dir = args.dcase2018_task2_dataset_dir
    workspace = args.workspace
    
    random_state = np.random.RandomState(1234)
    folds_num = 4
    
    # Paths
    csv_path = os.path.join(dcase2018_task2_dataset_dir, 'train.csv')
    
    # Read csv
    df = pd.DataFrame(pd.read_csv(csv_path))
    
    indexes = np.arange(len(df))
    random_state.shuffle(indexes)
    
    audios_num = len(df)
    audios_num_per_fold = int(audios_num // folds_num)

    # Create folds
    folds = np.zeros(audios_num, dtype=np.int32)
    
    for n in range(audios_num):
        folds[indexes[n]] = (n % folds_num) + 1

    df_ex = df
    df_ex['fold'] = folds

    # Write out validation csv
    out_path = os.path.join(workspace, 'dcase2018_task2_validate_meta.csv')
    df_ex.to_csv(out_path)

    print("Write out to {}".format(out_path))


def repeat_indexes(indexes, max_len, random_state):
    
    repeats_num = max_len // len(indexes) + 1
    
    repeated_indexes = []
    
    for n in range(repeats_num):
        random_state.shuffle(indexes)
        repeated_indexes.append(indexes)
        
    repeated_indexes = np.concatenate(repeated_indexes, axis=0)
    
    return repeated_indexes


def create_mixture_yaml(args):
    
    # Arguments & parameters
    dcase2018_task1_dataset_dir = args.dcase2018_task1_dataset_dir
    dcase2018_task2_dataset_dir = args.dcase2018_task2_dataset_dir
    
    workspace = args.workspace
    
    
    random_state = np.random.RandomState(1234)
    folds = [1, 2, 3, 4]
    mixed_audios_num = 2000
    events_per_clip = 3
    
    # Paths
    dcase2018_task1_meta = os.path.join(dcase2018_task1_dataset_dir, 'meta.csv')
    dcase2018_task2_meta = os.path.join(workspace, 'dcase2018_task2_validate_meta.csv')
    
    yaml_path = os.path.join(workspace, 'mixture.yaml')
    create_folder(os.path.dirname(yaml_path))
    
    # Scenes meta
    df_scenes = pd.read_csv(dcase2018_task1_meta, sep='\t')
    scene_names = np.array(df_scenes['filename'])
    random_state.shuffle(scene_names)

    # Events meta
    df_events = pd.read_csv(dcase2018_task2_meta, sep=',')
    
    
    # 
    count = 0
    data_list = []
    
    for fold in folds:
    
        bool_selected = (df_events['fold'] == fold) & (df_events['manually_verified'] == 1)
        event_audio_names = np.array(df_events.fname[bool_selected])
        event_labels = np.array(df_events.label[bool_selected])
        
        indexes = np.arange(len(event_audio_names))
        
        repeated_indexes = repeat_indexes(indexes, mixed_audios_num * events_per_clip, random_state)
        
    
        
        for n in range(mixed_audios_num):
            
            if count % 100 == 0:
                print(count)
            
            current_idxes = repeated_indexes[n * events_per_clip : (n + 1) * events_per_clip]
            
            events = []
            
            for (j, idx) in enumerate(current_idxes):
            
                event_audio_name = event_audio_names[idx]
                event_label = event_labels[idx]
                onset = j * 2.5 + 0.5
                
                audio_path = os.path.join(dcase2018_task2_dataset_dir, 'audio_train', event_audio_name)
                (audio, fs) = read_audio(audio_path)
                audio_duration = len(audio) / float(fs)
                audio_duration = min(audio_duration, 2.0)
                
                offset = onset + audio_duration
                
                events.append({'event_audio_name': event_audio_name, 
                            'event_label': event_label, 
                            'onset': onset, 
                            'offset': offset})
            
            scene_audio_name = scene_names[count].split('/')[1]
        
        
            data = {'mixture_name': '{:05d}.wav'.format(count), 
                    'fold': fold, 
                    'events': events, 
                    'scene_audio_name': scene_audio_name}
            data_list.append(data)
            
            count += 1
            
            # if count == 30:
            #     break
        
        
        with open(yaml_path, 'w') as f:
            f.write(yaml.dump(data_list, default_flow_style=False))
            # yaml.dump(data, f, default_flow_style=False)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('create_dcase2018_cv_folds')
    parser_a.add_argument('--dcase2018_task2_dataset_dir', type=str)
    parser_a.add_argument('--workspace', type=str)
    
    parser_b = subparsers.add_parser('create_mixture_yaml')
    parser_b.add_argument('--dcase2018_task1_dataset_dir', type=str)
    parser_b.add_argument('--dcase2018_task2_dataset_dir', type=str)
    parser_b.add_argument('--workspace', type=str)
    
    args = parser.parse_args()

    if args.mode == 'create_dcase2018_cv_folds':
        create_validation_folds(args)
        
    elif args.mode == 'create_mixture_yaml':
        add(args)
