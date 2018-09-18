import os
import sys
import numpy as np
import pandas as pd
import argparse
import random
import yaml
import pandas as pd

from utilities import create_folder, read_audio
import config


def create_dcase2018_task2_cross_validation_csv(args):
    """Create cross-validation csv and write to validate_meta.csv. 
    
    The created csv file looks like:
        ,fname,label,manually_verified,fold
        0,00044347.wav,Hi-hat,0,1
        1,001ca53d.wav,Saxophone,1,3
        ...
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


def repeat_array(array, max_len, random_state):
    """Shuffle and repeat an array to a given length. 
    
    Args:
      array: 1d-array. 
      max_len: int, maximum length of repeated array. 
      random_state: object. 
      
    Returns:
      repeated_array: 1d-array. 
    """
    
    repeats_num = max_len // len(array) + 1
    
    repeated_array = []
    
    for n in range(repeats_num):
        random_state.shuffle(array)
        repeated_array.append(array)
        
    repeated_array = np.concatenate(repeated_array, axis=0)
    repeated_array = repeated_array[0 : max_len]
    
    return repeated_array


def create_mixture_yaml(args):
    """Create mixture yaml file containing a list of information. Each 
    information looks like:
    
    - events:
    - event_audio_name: 19f45b13.wav
        event_label: Tambourine
        offset: 1.22
        onset: 0.5
    - event_audio_name: 63874688.wav
        event_label: Scissors
        offset: 3.38
        onset: 3.0
    - event_audio_name: cd3e20ec.wav
        event_label: Computer_keyboard
        offset: 7.5
        onset: 5.5
    fold: 1
    mixture_name: 00000.wav
    scene_audio_name: metro_station-barcelona-62-1861-a.wav
    """
    
    # Arguments & parameters
    dcase2018_task1_dataset_dir = args.dcase2018_task1_dataset_dir
    dcase2018_task2_dataset_dir = args.dcase2018_task2_dataset_dir
    
    workspace = args.workspace
    
    
    random_state = np.random.RandomState(1234)
    folds = [1, 2, 3, 4]
    mixed_audios_per_fold = 2000
    events_per_clip = 3
    total_events_per_fold = mixed_audios_per_fold * events_per_clip
    
    # Paths
    dcase2018_task1_meta = os.path.join(dcase2018_task1_dataset_dir, 
                                        'meta.csv')
    
    dcase2018_task2_meta = os.path.join(workspace, 
                                        'dcase2018_task2_validate_meta.csv')
    
    out_yaml_path = os.path.join(workspace, 'yaml_files', 'mixture.yaml')
    create_folder(os.path.dirname(out_yaml_path))
    
    # DCASE 2018 Task 1 acoutic scenes meta
    df_scenes = pd.read_csv(dcase2018_task1_meta, sep='\t')
    scene_names = np.array(df_scenes['filename'])
    random_state.shuffle(scene_names)

    # DCASE 2018 Task 2 sound events meta
    df_events = pd.read_csv(dcase2018_task2_meta, sep=',')
    events_audio_num = len(df_events)
    
    acoustic_scene_index = 0
    data_list = []

    # Calculate mixture meta
    for fold in folds:
    
        # Selected audios indexes
        bool_selected = (df_events['fold'] == fold) & \
                        (df_events['manually_verified'] == 1)
              
        selected_event_indexes = np.arange(events_audio_num)[bool_selected]
        
        repeated_event_indexes = repeat_array(array=selected_event_indexes, 
                                              max_len=total_events_per_fold, 
                                              random_state=random_state)

        for n in range(mixed_audios_per_fold):
            
            if acoustic_scene_index % 100 == 0:
                print('Fold {}, {} / {} mixture infos created'
                    ''.format(fold, acoustic_scene_index, 
                              mixed_audios_per_fold * len(folds)))
            
            event_indexes_for_one_clip = repeated_event_indexes[
                n * events_per_clip : (n + 1) * events_per_clip]
            
            events = []
            
            for j, index in enumerate(event_indexes_for_one_clip):
            
                event_audio_name = df_events.fname[index]
                event_label = df_events.label[index]
                onset = j * 2.5 + 0.5   # Onsets of events are 0.5 s, 3.0 s, 
                                        # 5.5 s in an audio clip. 
                
                event_audio_path = os.path.join(dcase2018_task2_dataset_dir, 
                                                'audio_train', event_audio_name)
                
                (audio, fs) = read_audio(event_audio_path)
                audio_duration = len(audio) / float(fs)
                audio_duration = min(audio_duration, 2.0)   # Clip maximum
                                                            # duration to 2.0 s.
                
                offset = onset + audio_duration
                
                events.append({'event_audio_name': event_audio_name, 
                               'event_label': event_label, 
                               'onset': onset, 
                               'offset': offset})
            
            scene_audio_name = scene_names[acoustic_scene_index].split('/')[1]
        
            data = {'mixture_name': '{:05d}.wav'.format(acoustic_scene_index), 
                    'fold': fold, 
                    'events': events, 
                    'scene_audio_name': scene_audio_name}
            data_list.append(data)
            
            acoustic_scene_index += 1
        
    # Write out yaml file
    with open(out_yaml_path, 'w') as f:
        yaml.dump(data_list, f, default_flow_style=False)
        
    print('Write out mixture yaml to {}'.format(out_yaml_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('create_dcase2018_task2_cross_validation_csv')
    parser_a.add_argument('--dcase2018_task2_dataset_dir', type=str)
    parser_a.add_argument('--workspace', type=str)
    
    parser_b = subparsers.add_parser('create_mixture_yaml')
    parser_b.add_argument('--dcase2018_task1_dataset_dir', type=str)
    parser_b.add_argument('--dcase2018_task2_dataset_dir', type=str)
    parser_b.add_argument('--workspace', type=str)
    
    args = parser.parse_args()

    if args.mode == 'create_dcase2018_task2_cross_validation_csv':
        create_dcase2018_task2_cross_validation_csv(args)
        
    elif args.mode == 'create_mixture_yaml':
        create_mixture_yaml(args)
