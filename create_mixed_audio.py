import os
import sys
import numpy as np
import pandas as pd
import argparse
import random
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging

from utilities import create_folder, read_audio, write_audio, calculate_average_energy, normalize_to_energy, create_logging, get_filename
import config


def create_mixed_audio(args):

    # Arguments & parameters
    dcase2018_task1_dataset_dir = args.dcase2018_task1_dataset_dir
    dcase2018_task2_dataset_dir = args.dcase2018_task2_dataset_dir
    workspace = args.workspace
    scene_type = args.scene_type
    snr = args.snr
    
    sample_rate = config.sample_rate
    clip_duration = 10.
    audio_len = int(sample_rate * clip_duration)
    random_state = np.random.RandomState(1234)
    
    # Paths
    yaml_path = os.path.join(workspace, 'yaml_files', 'mixture.yaml')
    
    out_audio_dir = os.path.join(workspace, 'mixed_audios', 'scene_type={},snr={}'.format(scene_type, snr))
    create_folder(out_audio_dir)

    with open(yaml_path, 'r') as f:
        data = yaml.load(f)
        
    create_audio_time = time.time()
        
    for n in range(len(data)):
    
        if n % 10 == 0:
            logging.info(n)
    
        if scene_type == 'white_noise':
            scene_audio = random_state.uniform(0., 1., audio_len)
            
        elif scene_type == 'dcase2018_task1':
            scene_audio_name = data[n]['scene_audio_name']
            scene_audio_path = os.path.join(dcase2018_task1_dataset_dir, 'audio', scene_audio_name)
            
            (scene_audio, fs) = read_audio(scene_audio_path, target_fs=sample_rate)
            
        # Normalize scene audio
        scene_audio = normalize_to_energy(scene_audio, db=-snr)
        
        # Reserve space
        events_audio = np.zeros(audio_len)
        
        for (j, event) in enumerate(data[n]['events']):
            
            audio_name = event['event_audio_name']
            onset = int(event['onset'] * sample_rate)
            offset = int(event['offset'] * sample_rate)
            
            audio_path = os.path.join(dcase2018_task2_dataset_dir, 'audio_train', audio_name)
            (audio, fs) = read_audio(audio_path, target_fs=sample_rate)
            
            audio = normalize_to_energy(audio, db=0.)
            
            events_audio[onset : offset] = audio[0 : offset - onset]
            
        stereo_audio = np.array((events_audio, scene_audio)).T
        stereo_audio /= np.max(np.abs(stereo_audio))
            
        out_audio_path = os.path.join(out_audio_dir, data[n]['mixture_name'])
        write_audio(out_audio_path, stereo_audio, sample_rate)
        
    logging.info('Write out audio finished! {} s'.format(time.time() - create_audio_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dcase2018_task1_dataset_dir', type=str)
    parser.add_argument('--dcase2018_task2_dataset_dir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--scene_type', type=str, choices=['white_noise', 'dcase2018_task1'], required=True)
    parser.add_argument('--snr', type=int, required=True)

    args = parser.parse_args()
    
    logs_dir = os.path.join(args.workspace, 'logs', get_filename(__file__))
    create_folder(logs_dir)
    logging = create_logging(logs_dir, filemode='w')
    
    logging.info(args)
    
    create_mixed_audio(args)