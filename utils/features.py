import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv
import random
import yaml
import logging

from utilities import read_audio, create_folder, get_filename, create_logging
import config


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=50., 
                                        fmax=sample_rate // 2).T
        '''(fft_size, mel_bins)'''
    
    def transform(self, audio):
    
        x = self.transform_stft(audio)
    
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x
        
    def transform_stft(self, audio):
        
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win, 
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude')
        
        x = x.T
        x = x.astype(np.float32)
        
        return x
        
    def get_inverse_melW(self):
        """Transformation matrix for convert back from mel bins to stft bins. 
        """
        
        W = self.melW.T     # (mel_bins, fft_size)
        invW = W / (np.sum(W, axis=0) + 1e-8)
        return invW


def calculate_logmel(audio_path, sample_rate, feature_extractor):
    
    # Read audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate, mono=False)
    
    events_audio = audio[:, 0]
    scene_audio = audio[:, 1]
    mixed_audio = np.mean(audio, axis=-1)
    
    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''
    
    # Extract feature
    mixture_logmel = feature_extractor.transform(mixed_audio)
    mixture_stft = feature_extractor.transform_stft(mixed_audio)
    events_stft = feature_extractor.transform_stft(events_audio)
    scene_stft = feature_extractor.transform_stft(scene_audio)
    
    dict = {'mixture_logmel': mixture_logmel, 
            'mixture_stft': mixture_stft, 
            'events_stft': events_stft, 
            'scene_stft': scene_stft}
    
    return dict


def read_development_meta(meta_csv):
    
    df = pd.read_csv(meta_csv, sep='\t')
    df = pd.DataFrame(df)
    
    audio_names = []
    scene_labels = []
    identifiers = []
    source_labels = []
    
    for row in df.iterrows():
        
        audio_name = row[1]['filename'].split('/')[1]
        scene_label = row[1]['scene_label']
        identifier = row[1]['identifier']
        source_label = row[1]['source_label']
        
        audio_names.append(audio_name)
        scene_labels.append(scene_label)
        identifiers.append(identifier)
        source_labels.append(source_label)
        
    return audio_names, scene_labels, identifiers, source_labels
    
    
def read_evaluation_meta(evaluation_csv):
    
    with open(evaluation_csv, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    audio_names = []
        
    for li in lis:
        audio_name = li[0].split('/')[1]
        audio_names.append(audio_name)
        
    return audio_names
    

def get_target_from_events(events, lb_to_ix):
    
    classes_num = len(lb_to_ix)
    target = np.zeros(classes_num, dtype=np.int32)
    
    for event in events:
        ix = lb_to_ix[event['event_label']]
        target[ix] = 1
        
    return target


def calculate_logmel_features(args):
    
    # Arguments & parameters
    workspace = args.workspace
    scene_type = args.scene_type
    snr = args.snr

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    stft_bins = window_size // 2 + 1
    classes_num = len(config.labels)
    lb_to_ix = config.lb_to_ix
    
    # Paths
    audio_dir = os.path.join(workspace, 'mixed_audios', 
                             'scene_type={},snr={}'.format(scene_type, snr))
    
    yaml_path = os.path.join(workspace, 'yaml_files', 'mixture.yaml')
    
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 
        'scene_type={},snr={}'.format(scene_type, snr), 'development.h5')
        
    create_folder(os.path.dirname(hdf5_path))

    # Load mixture yaml
    load_time = time.time()
    
    with open(yaml_path, 'r') as f:
        data_list = yaml.load(f)
        
    logging.info('Loading mixture yaml time: {} s'
        ''.format(time.time() - load_time))
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)

    # Create hdf5 file
    write_hdf5_time = time.time()
    
    hf = h5py.File(hdf5_path, 'w')
    
    hf.create_dataset(
        name='mixture_logmel', 
        shape=(0, seq_len, mel_bins), 
        maxshape=(None, seq_len, mel_bins), 
        dtype=np.float32)
        
    hf.create_dataset(
        name='mixture_stft', 
        shape=(0, seq_len, stft_bins), 
        maxshape=(None, seq_len, stft_bins), 
        dtype=np.float32)
        
    hf.create_dataset(
        name='events_stft', 
        shape=(0, seq_len, stft_bins), 
        maxshape=(None, seq_len, stft_bins), 
        dtype=np.float32)
        
    hf.create_dataset(
        name='scene_stft', 
        shape=(0, seq_len, stft_bins), 
        maxshape=(None, seq_len, stft_bins), 
        dtype=np.float32)
        
    hf.create_dataset(
        name='target', 
        shape=(0, classes_num), 
        maxshape=(None, classes_num), 
        dtype=np.int32)
        
    mixture_names = []
    
    folds = []

    for n, data in enumerate(data_list):
        
        if n % 10 == 0:
            logging.info('{} / {} audio features calculated'
                ''.format(n, len(data_list)))
            
        mixed_audio_name = data['mixture_name']
        mixed_audio_path = os.path.join(audio_dir, mixed_audio_name)
    
        mixture_names.append(data['mixture_name'])
        folds.append(data['fold'])
    
        # Extract feature
        features_dict = calculate_logmel(audio_path=mixed_audio_path, 
                                         sample_rate=sample_rate, 
                                         feature_extractor=feature_extractor)
    
        # Write out features
        hf['mixture_logmel'].resize((n + 1, seq_len, mel_bins))
        hf['mixture_logmel'][n] = features_dict['mixture_logmel']

        hf['mixture_stft'].resize((n + 1, seq_len, stft_bins))
        hf['mixture_stft'][n] = features_dict['mixture_stft']
        
        hf['events_stft'].resize((n + 1, seq_len, stft_bins))
        hf['events_stft'][n] = features_dict['events_stft']
        
        hf['scene_stft'].resize((n + 1, seq_len, stft_bins))
        hf['scene_stft'][n] = features_dict['scene_stft']
    
        # Write out target
        target = get_target_from_events(data['events'], lb_to_ix)
        hf['target'].resize((n + 1, classes_num))
        hf['target'][n] = target
    
    hf.create_dataset(name='audio_name', 
                      data=[s.encode() for s in mixture_names], 
                      dtype='S20')
                        
    hf.create_dataset(name='fold', 
                      data=folds, 
                      dtype=np.int32)

    hf.close()
    
    logging.info('Write out hdf5 file to {}'.format(hdf5_path))
    logging.info('Time spent: {} s'.format(time.time() - write_hdf5_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--scene_type', type=str, required=True)
    parser_logmel.add_argument('--snr', type=int, required=True)
    
    args = parser.parse_args()
    
    logs_dir = os.path.join(args.workspace, 'logs', get_filename(__file__))
    create_folder(logs_dir)
    logging = create_logging(logs_dir, filemode='w')
    
    logging.info(args)
    
    if args.mode == 'logmel':
        calculate_logmel_features(args)
        
    else:
        raise Exception('Incorrect arguments!')