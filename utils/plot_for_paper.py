import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import yaml
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import DataGenerator, InferenceDataGenerator
from main_pytorch import get_model
from utilities import *
from models_pytorch import *
import config
from features import LogMelExtractor
from get_avg_stats import get_est_event_list, get_ref_event_list


def plot_waveform(args):
    
    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    scene_type = args.scene_type
    snr = args.snr
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size-overlap
    mel_bins = config.mel_bins
    seq_len = config.seq_len
    ix_to_lb = config.ix_to_lb
    
    thres = 0.1
    batch_size = 24
    
    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'scene_type={},snr={}'.format(scene_type, snr), 'development.h5')

    yaml_path = os.path.join(workspace, 'yaml_files', 'mixture.yaml')
    
    audios_dir = os.path.join(workspace, 'mixed_audios', 'scene_type={},snr={}'.format(scene_type, snr))
    
    # Load yaml file
    load_yaml_time = time.time()
    
    with open(yaml_path, 'r') as f:
        meta = yaml.load(f)
        
    print('Load yaml file time: {:.3f} s'.format(time.time() - load_yaml_time))

    # Data generator
    generator = InferenceDataGenerator(hdf5_path=hdf5_path,
                            batch_size=batch_size, 
                            holdout_fold=holdout_fold)

    generate_func = generator.generate_validate(data_type='validate', 
                                                shuffle=False, 
                                                max_iteration=None)
    
    
    # Evaluate on mini-batch
    for (iteration, data) in enumerate(generate_func):
        
        logging.info(iteration)
        
        (batch_x, batch_y, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        batch_gt_masks = []
        batch_single_gt_masks = []
        batch_mixture_stfts = []

        for n in range(len(batch_audio_names)):
            curr_meta = search_meta_by_mixture_name(meta, batch_audio_names[n])
            curr_events = curr_meta['events']
              
            gt_indexes = get_ground_truth_indexes(curr_events)
 
            gt_sed = get_sed_from_meta(curr_events) # (seq_len, classes_num)
              
            (events_stft, scene_stft, mixture_stft) = generator.get_events_scene_mixture_stft(batch_audio_names[n])

            gt_mask = ideal_ratio_mask(events_stft, scene_stft)    # (seq_len, fft_size)
            
            gt_masks = gt_mask[:, :, None] * gt_sed[:, None, :] # (seq_len, fft_size, classes_num)
            gt_masks = gt_masks.astype(np.float32)
            batch_gt_masks.append(gt_masks)
            batch_single_gt_masks.append(gt_mask)
            
            batch_mixture_stfts.append(mixture_stft)
            
        # Plot waveform & spectrogram & ideal ratio mask
        if True:
            for n in range(len(batch_x)):

                print(batch_audio_names[n])
                print(batch_y[n])
                target_labels = target_to_labels(batch_y[n], labels)
                print(target_labels)
                
                mixed_audio_path = os.path.join(audios_dir, batch_audio_names[n])
                (mixed_audio, _) = read_audio(mixed_audio_path, target_fs=config.sample_rate, mono=True)
                mixed_audio /= np.max(np.abs(mixed_audio))
                
                fig, axs = plt.subplots(3, 1, figsize=(6, 6))
                
                axs[0].plot(mixed_audio)
                axs[0].set_title('Waveform')
                axs[0].xaxis.set_ticks([0, len(mixed_audio)])
                axs[0].xaxis.set_ticklabels(['0.0', '10.0 s'])
                axs[0].set_xlim(0, len(mixed_audio))
                axs[0].set_ylim(-1, 1)
                axs[0].set_xlabel('time')
                axs[0].set_ylabel('Amplitude')
                
                axs[1].matshow(np.log(batch_mixture_stfts[n]).T, origin='lower', aspect='auto', cmap='jet')
                axs[1].set_title('Spectrogram')
                axs[1].xaxis.set_ticks([0, 310])
                axs[1].xaxis.set_ticklabels(['0.0', '10.0 s'])
                axs[1].xaxis.tick_bottom()
                axs[1].yaxis.set_ticks([0, 1024])
                axs[1].yaxis.set_ticklabels(['0', '1025'])
                axs[1].set_xlabel('time')
                axs[1].set_ylabel('FFT bins')
                
                axs[2].matshow(batch_single_gt_masks[n].T, origin='lower', aspect='auto', cmap='jet')
                axs[2].set_title('Ideal ratio mask')
                axs[2].xaxis.set_ticks([0, 310])
                axs[2].xaxis.set_ticklabels(['0.0', '10.0 s'])
                axs[2].xaxis.tick_bottom()
                axs[2].yaxis.set_ticks([0, 1024])
                axs[2].yaxis.set_ticklabels(['0', '1025'])
                axs[2].set_xlabel('time')
                axs[2].set_ylabel('FFT bins')
                
                plt.tight_layout()
                plt.show()
                return
                    
                    
def plot_mel_masks(args):
    
    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    scene_type = args.scene_type
    snr = args.snr
    iteration = args.iteration
    model_type = args.model_type
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size-overlap
    mel_bins = config.mel_bins
    seq_len = config.seq_len
    ix_to_lb = config.ix_to_lb
    
    thres = 0.1
    batch_size = 24

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 
        'scene_type={},snr={}'.format(scene_type, snr), 'development.h5')

    model_path = os.path.join(workspace, 'models', 'main_pytorch', 
        'model_type={}'.format(model_type), 'scene_type={},snr={}'
        ''.format(scene_type, snr), 'holdout_fold{}'.format(holdout_fold), 
        'md_{}_iters.tar'.format(iteration))
    
    yaml_path = os.path.join(workspace, 'yaml_files', 'mixture.yaml')
    
    audios_dir = os.path.join(workspace, 'mixed_audios', 
                              'scene_type={},snr={}'.format(scene_type, snr))
    
    sep_wavs_dir = os.path.join(workspace, 'separated_wavs', 'main_pytorch', 
        'model_type={}'.format(model_type), 
        'scene_type={},snr={}'.format(scene_type, snr), 
        'holdout_fold{}'.format(holdout_fold))
        
    create_folder(sep_wavs_dir)
    
    '''
    t1 = time.time()
    with open(yaml_path, 'r') as f:
        data = yaml.load(f)
    print(time.time() - t1)
    
    import pickle
    create_folder(os.path.join(workspace, '_tmp'))
    pickle.dump(data, open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'wb'))
    '''
    
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    
    inverse_melW = feature_extractor.get_inverse_melW()
    
    meta = pickle.load(open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'rb'))
    
    # Load model
    Model = get_model(model_type)
    model = Model(classes_num, seq_len, mel_bins, cuda)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = InferenceDataGenerator(hdf5_path=hdf5_path,
                            batch_size=batch_size, 
                            holdout_fold=holdout_fold)

    generate_func = generator.generate_validate(data_type='validate', 
                                                    shuffle=False, 
                                                    max_iteration=None)
    
    
    # Evaluate on mini-batch
    for (iteration, data) in enumerate(generate_func):
        
        logging.info(iteration)
        
        (batch_x, batch_y, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        
        (batch_output, batch_bottleneck) = model(batch_x, return_bottleneck=True)
    
        batch_output = batch_output.data.cpu().numpy()
        batch_bottleneck = batch_bottleneck.data.cpu().numpy()  # (batch_size, classes_num, seq_len, mel_bins)

        batch_pred_sed = np.mean(batch_bottleneck, axis=-1)
        batch_pred_sed = np.transpose(batch_pred_sed, (0, 2, 1))    # (batch_size, seq_len, classes_num)
        
        batch_gt_masks = []
        
        for n in range(len(batch_audio_names)):
            curr_meta = search_meta_by_mixture_name(meta, batch_audio_names[n])
            curr_events = curr_meta['events']
              
            pred_indexes = np.where(batch_output[n] > thres)[0]
            gt_indexes = get_ground_truth_indexes(curr_events)
 
            gt_sed = get_sed_from_meta(curr_events) # (seq_len, classes_num)
            
            pred_sed = np.zeros((seq_len, classes_num))
            pred_sed[:, pred_indexes] = batch_pred_sed[n][:, pred_indexes]  # (seq_len, classes_num)
 
            (events_stft, scene_stft, _) = generator.get_events_scene_mixture_stft(batch_audio_names[n])
            events_stft = np.dot(events_stft, feature_extractor.melW)
            scene_stft = np.dot(scene_stft, feature_extractor.melW)
            
            gt_mask = ideal_binary_mask(events_stft, scene_stft)    # (seq_len, fft_size)
            
            gt_masks = gt_mask[:, :, None] * gt_sed[:, None, :] # (seq_len, fft_size, classes_num)
            gt_masks = gt_masks.astype(np.float32)
            batch_gt_masks.append(gt_masks)
            
            pred_masks = batch_bottleneck[n].transpose(1, 2, 0) # (seq_len, fft_size, classes_num)

            if False:
                curr_audio_name = curr_meta['mixture_name']
                audio_path = os.path.join(audios_dir, curr_audio_name)
                (mixed_audio, fs) = read_audio(audio_path, target_fs=sample_rate, mono=True)
                
                out_wav_path = os.path.join(sep_wavs_dir, curr_audio_name)
                write_audio(out_wav_path, mixed_audio, sample_rate)
                
                window = np.hamming(window_size)
                mixed_stft_cmplx = stft(x=mixed_audio, window_size=window_size, hop_size=hop_size, window=window, mode='complex')
                mixed_stft_cmplx = mixed_stft_cmplx[0 : seq_len, :]
                mixed_stft = np.abs(mixed_stft_cmplx)
                
                for k in gt_indexes:
                    masked_stft = np.dot(pred_masks[:, :, k], inverse_melW) * mixed_stft
                    masked_stft_cmplx = real_to_complex(masked_stft, mixed_stft_cmplx)
                    
                    frames = istft(masked_stft_cmplx)
                    cola_constant = get_cola_constant(hop_size, window)
                    sep_audio = overlap_add(frames, hop_size, cola_constant)
                    
                    sep_wav_path = os.path.join(sep_wavs_dir, '{}_{}.wav'.format(os.path.splitext(curr_audio_name)[0], ix_to_lb[k]))
                    write_audio(sep_wav_path, sep_audio, sample_rate)
                    logging.info('Audio wrote to {}'.format(sep_wav_path))
      
            
        if True:
            for n in range(len(batch_output)):
            
                # if batch_audio_names[n] == '00013.wav':
                if True:
            
                    (events_stft, scene_stft, _) = generator.get_events_scene_mixture_stft(batch_audio_names[n])
                    
                    tmp = events_stft / (events_stft + scene_stft)
                    # tmp = add(events_stft, scene_stft)
                
                    
                    print(batch_audio_names[n])
                    print(batch_y[n])
                    target_labels = target_to_labels(batch_y[n], labels)
                    print(target_labels)

                    fig, axs = plt.subplots(7, 7, figsize=(15, 10))
                    for k in range(classes_num):
                        axs[k // 6, k % 6].matshow(batch_bottleneck[n, k].T, origin='lower', aspect='auto', cmap='jet')
                        if labels[k] in target_labels:
                            color = 'r'
                        else:
                            color = 'k'
                        axs[k // 6, k % 6].set_title(labels[k], color=color)
                        axs[k // 6, k % 6].xaxis.set_ticks([])
                        axs[k // 6, k % 6].yaxis.set_ticks([])
                        axs[k // 6, k % 6].set_xlabel('time')
                        axs[k // 6, k % 6].set_ylabel('mel bins')
                        
                    axs[6, 5].matshow(np.log(events_stft).T, origin='lower', aspect='auto', cmap='jet')
                    axs[6, 5].set_title('Spectrogram')
                    axs[6, 5].xaxis.set_ticks([0, 310])
                    axs[6, 5].xaxis.set_ticklabels(['0.0', '10.0 s'])
                    axs[6, 5].xaxis.tick_bottom()
                    axs[6, 5].yaxis.set_ticks([0, 1024])
                    axs[6, 5].yaxis.set_ticklabels(['0', '1025'])
                    axs[6, 5].set_xlabel('time')
                    axs[6, 5].set_ylabel('FFT bins')
                    
                    axs[6, 6].matshow(np.log(np.dot(events_stft, feature_extractor.melW)).T, origin='lower', aspect='auto', cmap='jet')
                    axs[6, 5].set_title('Log mel pectrogram')
                    axs[6, 6].xaxis.set_ticks([0, 310])
                    axs[6, 6].xaxis.set_ticklabels(['0.0', '10.0 s'])
                    axs[6, 6].xaxis.tick_bottom()
                    axs[6, 6].yaxis.set_ticks([0, 63])
                    axs[6, 6].yaxis.set_ticklabels(['0', '64'])
                    axs[6, 6].set_xlabel('time')
                    axs[6, 6].set_ylabel('mel bins')
                    
                    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
                    plt.show()
                    
                    # frame SED
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    score_mat = []
                    for k in range(classes_num):
                        score = np.mean(batch_bottleneck[n, k], axis=-1)
                        score_mat.append(score)
                        
                    score_mat = np.array(score_mat)
                    
                    ax.matshow(score_mat, origin='lower', aspect='auto', cmap='jet')
                    # ax.set_title('Frame-wise predictions')
                    ax.xaxis.set_ticks([0, 310])
                    ax.xaxis.set_ticklabels(['0.0', '10.0 s'])
                    ax.xaxis.tick_bottom()
                    ax.set_xlabel('time')
                    ax.yaxis.set_ticks(np.arange(classes_num))
                    ax.yaxis.set_ticklabels(config.labels, fontsize='xx-small')
                    ax.yaxis.grid(color='k', linestyle='solid', linewidth=0.3)
                    
                    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
                    plt.show()
                    
                    # event SED
                    est_event_list = get_est_event_list(batch_pred_sed[n:n+1], batch_audio_names[n:n+1], labels)
                    event_mat = event_list_to_matrix(est_event_list)
                    
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    ax.matshow(event_mat.T, origin='lower', aspect='auto', cmap='jet')
                    # ax.set_title('Event-wise predictions')
                    ax.xaxis.set_ticks([0, 310])
                    ax.xaxis.set_ticklabels(['0.0', '10.0 s'])
                    ax.xaxis.tick_bottom()
                    ax.set_xlabel('time')
                    ax.yaxis.set_ticks(np.arange(classes_num))
                    ax.yaxis.set_ticklabels(config.labels, fontsize='xx-small')
                    ax.yaxis.grid(color='k', linestyle='solid', linewidth=0.3)
                    
                    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
                    plt.show()
                    
                    # gt SED
                    ref_event_list = get_ref_event_list(meta, batch_audio_names[n:n+1])
                    event_mat = event_list_to_matrix(ref_event_list)
                    
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    ax.matshow(event_mat.T, origin='lower', aspect='auto', cmap='jet')
                    # ax.set_title('Event-wise predictions')
                    ax.xaxis.set_ticks([0, 310])
                    ax.xaxis.set_ticklabels(['0.0', '10.0 s'])
                    ax.xaxis.tick_bottom()
                    ax.set_xlabel('time')
                    ax.yaxis.set_ticks(np.arange(classes_num))
                    ax.yaxis.set_ticklabels(config.labels, fontsize='xx-small')
                    ax.yaxis.grid(color='k', linestyle='solid', linewidth=0.3)
                    
                    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
                    plt.show()
                    
                    return
                    

def event_list_to_matrix(event_list):
    
    lb_to_ix = config.lb_to_ix
    hop_size = config.window_size - config.overlap
    frames_per_second = config.sample_rate / float(hop_size)
    
    mat = np.zeros((config.seq_len, len(config.labels)))
    
    for event in event_list:
        onset = int(event['onset'] * frames_per_second) + 1
        offset = int(event['offset'] * frames_per_second) + 1
        event_label = event['event_label']
        ix = lb_to_ix[event_label]
        mat[onset : offset, ix] = 1
        
    return mat


def plot_framework(args):
    
    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    scene_type = args.scene_type
    snr = args.snr
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size-overlap
    mel_bins = config.mel_bins
    seq_len = config.seq_len
    ix_to_lb = config.ix_to_lb
    
    thres = 0.1
    batch_size = 24
    
    
    if filename == 'tmp01b':
        Model = Vggish_postsigmoid
        model = Model(classes_num)
    
    elif filename == 'tmp02b':
        Model = VggishGAP_postsigmoid
        model = Model(classes_num)
    
    elif filename == 'tmp03b':
        Model = VggishGWRP_postsigmoid
        model = Model(classes_num, seq_len, mel_bins, cuda)
  
    else:
        raise Exception('Error!')

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'scene_type={},snr={}'.format(scene_type, snr), 'development.h5')

    model_path = os.path.join(workspace, 'models', filename, 'scene_type={},snr={}'.format(scene_type, snr), 'holdout_fold{}'.format(holdout_fold), 'md_{}_iters.tar'.format(iteration))
    
    yaml_path = os.path.join(workspace, 'yaml_files', 'mixture.yaml')
    
    audios_dir = os.path.join(workspace, 'mixed_audios', 'scene_type={},snr={}'.format(scene_type, snr))
    
    sep_wavs_dir = os.path.join(workspace, 'separated_wavs', filename, 'scene_type={},snr={}'.format(scene_type, snr), 'holdout_fold{}'.format(holdout_fold))
    create_folder(sep_wavs_dir)
    
    stat_path = os.path.join(workspace, 'stats', filename, 'scene_type={},snr={}'.format(scene_type, snr), 'holdout_fold{}'.format(holdout_fold), 'stat.p')
    create_folder(os.path.dirname(stat_path))
    
    '''
    t1 = time.time()
    with open(yaml_path, 'r') as f:
        data = yaml.load(f)
    print(time.time() - t1)
    
    import pickle
    create_folder(os.path.join(workspace, '_tmp'))
    pickle.dump(data, open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'wb'))
    '''
    
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    
    inverse_melW = feature_extractor.get_inverse_melW()
    
    
    meta = pickle.load(open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'rb'))
    
    # Load model
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = InferenceDataGenerator(hdf5_path=hdf5_path,
                            batch_size=batch_size, 
                            holdout_fold=holdout_fold)

    generate_func = generator.generate_validate(data_type='validate', 
                                                    shuffle=False, 
                                                    max_iteration=None)
    
    
    # Evaluate on mini-batch
    for (iteration, data) in enumerate(generate_func):
        
        logging.info(iteration)
        
        (batch_x, batch_y, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        
        (batch_output, batch_bottleneck) = model(batch_x, return_bottleneck=True)
    
        batch_output = batch_output.data.cpu().numpy()
        batch_bottleneck = batch_bottleneck.data.cpu().numpy()  # (batch_size, classes_num, seq_len, mel_bins)

        batch_pred_sed = np.mean(batch_bottleneck, axis=-1)
        batch_pred_sed = np.transpose(batch_pred_sed, (0, 2, 1))    # (batch_size, seq_len, classes_num)
        
        batch_gt_masks = []
        
        for n in range(len(batch_audio_names)):
            curr_meta = search_meta_by_mixture_name(meta, batch_audio_names[n])
            curr_events = curr_meta['events']
              
            pred_indexes = np.where(batch_output[n] > thres)[0]
            gt_indexes = get_ground_truth_indexes(curr_events)
 
            gt_sed = get_sed_from_meta(curr_events) # (seq_len, classes_num)
            
            pred_sed = np.zeros((seq_len, classes_num))
            pred_sed[:, pred_indexes] = batch_pred_sed[n][:, pred_indexes]  # (seq_len, classes_num)
 
            (events_stft, scene_stft, _) = generator.get_events_scene_mixture_stft(batch_audio_names[n])
            events_stft = np.dot(events_stft, feature_extractor.melW)
            scene_stft = np.dot(scene_stft, feature_extractor.melW)
            
            gt_mask = ideal_binary_mask(events_stft, scene_stft)    # (seq_len, fft_size)
            
            gt_masks = gt_mask[:, :, None] * gt_sed[:, None, :] # (seq_len, fft_size, classes_num)
            gt_masks = gt_masks.astype(np.float32)
            batch_gt_masks.append(gt_masks)
            
            pred_masks = batch_bottleneck[n].transpose(1, 2, 0) # (seq_len, fft_size, classes_num)

            if False:
                curr_audio_name = curr_meta['mixture_name']
                audio_path = os.path.join(audios_dir, curr_audio_name)
                (mixed_audio, fs) = read_audio(audio_path, target_fs=sample_rate, mono=True)
                
                out_wav_path = os.path.join(sep_wavs_dir, curr_audio_name)
                write_audio(out_wav_path, mixed_audio, sample_rate)
                
                window = np.hamming(window_size)
                mixed_stft_cmplx = stft(x=mixed_audio, window_size=window_size, hop_size=hop_size, window=window, mode='complex')
                mixed_stft_cmplx = mixed_stft_cmplx[0 : seq_len, :]
                mixed_stft = np.abs(mixed_stft_cmplx)
                
                for k in gt_indexes:
                    masked_stft = np.dot(pred_masks[:, :, k], inverse_melW) * mixed_stft
                    masked_stft_cmplx = real_to_complex(masked_stft, mixed_stft_cmplx)
                    
                    frames = istft(masked_stft_cmplx)
                    cola_constant = get_cola_constant(hop_size, window)
                    sep_audio = overlap_add(frames, hop_size, cola_constant)
                    
                    sep_wav_path = os.path.join(sep_wavs_dir, '{}_{}.wav'.format(os.path.splitext(curr_audio_name)[0], ix_to_lb[k]))
                    write_audio(sep_wav_path, sep_audio, sample_rate)
                    logging.info('Audio wrote to {}'.format(sep_wav_path))
      
            
        if True:
            for n in range(len(batch_output)):
            
                # if batch_audio_names[n] == '00013.wav':
                if True:
            
                    (events_stft, scene_stft, mixture_stft) = generator.get_events_scene_mixture_stft(batch_audio_names[n])
                    
                    tmp = events_stft / (events_stft + scene_stft)
                    # tmp = add(events_stft, scene_stft)
                
                    
                    print(batch_audio_names[n])
                    print(batch_y[n])
                    target_labels = target_to_labels(batch_y[n], labels)
                    print(target_labels)
                    
                    fig, axs = plt.subplots(7, 7, figsize=(15, 10))
                    for k in range(classes_num):
                        sep_stft = mixture_stft * np.dot(batch_bottleneck[n, k], inverse_melW)
                        axs[k // 6, k % 6].matshow(np.log(sep_stft).T, origin='lower', aspect='auto', cmap='jet', vmin=-15, vmax=0.)
                        if labels[k] in target_labels:
                            color = 'r'
                        else:
                            color = 'k'
                        axs[k // 6, k % 6].set_title(labels[k], color=color)
                        axs[k // 6, k % 6].xaxis.set_ticks([])
                        axs[k // 6, k % 6].yaxis.set_ticks([])
                        axs[k // 6, k % 6].set_xlabel('time')
                        axs[k // 6, k % 6].set_ylabel('FFT bins')
                        
                    axs[6, 5].matshow(np.log(mixture_stft).T, origin='lower', aspect='auto', cmap='jet', vmin=-15, vmax=0.)
                    axs[6, 5].xaxis.set_ticks([0, 310])
                    axs[6, 5].xaxis.set_ticklabels(['0.0', '10.0 s'])
                    axs[6, 5].xaxis.tick_bottom()
                    axs[6, 5].yaxis.set_ticks([0, 1024])
                    axs[6, 5].yaxis.set_ticklabels(['0', '1025'])
                    axs[6, 5].set_xlabel('time')
                    axs[6, 5].set_ylabel('FFT bins')
                    
                    axs[6, 6].matshow(np.log(np.dot(events_stft, feature_extractor.melW)).T, origin='lower', aspect='auto', cmap='jet')
                    axs[6, 6].xaxis.set_ticks([0, 310])
                    axs[6, 6].xaxis.set_ticklabels(['0.0', '10.0 s'])
                    axs[6, 6].xaxis.tick_bottom()
                    axs[6, 6].yaxis.set_ticks([0, 63])
                    axs[6, 6].yaxis.set_ticklabels(['0', '64'])
                    axs[6, 6].set_xlabel('time')
                    axs[6, 6].set_ylabel('mel bins')
                        
                    # axs[6, 6].matshow(tmp.T, origin='lower', aspect='auto', cmap='jet')
                    # axs[6, 5].matshow(tmp.T, origin='lower', aspect='auto', cmap='jet')
                    # axs[6, 5].matshow(np.log(events_stft + scene_stft).T, origin='lower', aspect='auto', cmap='jet')
                        
                    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
                    # plt.tight_layout()
                    plt.show()
                    # return




def add(args):
    
    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    scene_type = args.scene_type
    snr = args.snr
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size-overlap
    mel_bins = config.mel_bins
    seq_len = config.seq_len
    ix_to_lb = config.ix_to_lb
    
    thres = 0.1
    batch_size = 24
    
    
    if filename == 'tmp01b':
        Model = Vggish_postsigmoid
        model = Model(classes_num)
    
    elif filename == 'tmp02b':
        Model = VggishGAP_postsigmoid
        model = Model(classes_num)
    
    elif filename == 'tmp03b':
        Model = VggishGWRP_postsigmoid
        model = Model(classes_num, seq_len, mel_bins, cuda)
  
    else:
        raise Exception('Error!')

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'scene_type={},snr={}'.format(scene_type, snr), 'development.h5')

    model_path = os.path.join(workspace, 'models', filename, 'scene_type={},snr={}'.format(scene_type, snr), 'holdout_fold{}'.format(holdout_fold), 'md_{}_iters.tar'.format(iteration))
    
    yaml_path = os.path.join(workspace, 'yaml_files', 'mixture.yaml')
    
    audios_dir = os.path.join(workspace, 'mixed_audios', 'scene_type={},snr={}'.format(scene_type, snr))
    
    sep_wavs_dir = os.path.join(workspace, 'separated_wavs', filename, 'scene_type={},snr={}'.format(scene_type, snr), 'holdout_fold{}'.format(holdout_fold))
    create_folder(sep_wavs_dir)
    
    stat_path = os.path.join(workspace, 'stats', filename, 'scene_type={},snr={}'.format(scene_type, snr), 'holdout_fold{}'.format(holdout_fold), 'stat.p')
    create_folder(os.path.dirname(stat_path))
    
    '''
    t1 = time.time()
    with open(yaml_path, 'r') as f:
        data = yaml.load(f)
    print(time.time() - t1)
    
    import pickle
    create_folder(os.path.join(workspace, '_tmp'))
    pickle.dump(data, open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'wb'))
    '''
    
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    
    inverse_melW = feature_extractor.get_inverse_melW()
    
    
    meta = pickle.load(open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'rb'))
    
    # Load model
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = InferenceDataGenerator(hdf5_path=hdf5_path,
                            batch_size=batch_size, 
                            holdout_fold=holdout_fold)

    generate_func = generator.generate_validate(data_type='validate', 
                                                    shuffle=False, 
                                                    max_iteration=None)
    
    
    # Evaluate on mini-batch
    for (iteration, data) in enumerate(generate_func):
        
        logging.info(iteration)
        
        (batch_x, batch_y, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        
        (batch_output, batch_bottleneck) = model(batch_x, return_bottleneck=True)
    
        batch_output = batch_output.data.cpu().numpy()
        batch_bottleneck = batch_bottleneck.data.cpu().numpy()  # (batch_size, classes_num, seq_len, mel_bins)

        batch_pred_sed = np.mean(batch_bottleneck, axis=-1)
        batch_pred_sed = np.transpose(batch_pred_sed, (0, 2, 1))    # (batch_size, seq_len, classes_num)
        
        batch_gt_masks = []
        
        for n in range(len(batch_audio_names)):
            curr_meta = search_meta_by_mixture_name(meta, batch_audio_names[n])
            curr_events = curr_meta['events']
              
            pred_indexes = np.where(batch_output[n] > thres)[0]
            gt_indexes = get_ground_truth_indexes(curr_events)
 
            gt_sed = get_sed_from_meta(curr_events) # (seq_len, classes_num)
            
            pred_sed = np.zeros((seq_len, classes_num))
            pred_sed[:, pred_indexes] = batch_pred_sed[n][:, pred_indexes]  # (seq_len, classes_num)
 
            (events_stft, scene_stft, _) = generator.get_events_scene_mixture_stft(batch_audio_names[n])
            events_stft = np.dot(events_stft, feature_extractor.melW)
            scene_stft = np.dot(scene_stft, feature_extractor.melW)
            
            gt_mask = ideal_binary_mask(events_stft, scene_stft)    # (seq_len, fft_size)
            
            gt_masks = gt_mask[:, :, None] * gt_sed[:, None, :] # (seq_len, fft_size, classes_num)
            gt_masks = gt_masks.astype(np.float32)
            batch_gt_masks.append(gt_masks)
            
            pred_masks = batch_bottleneck[n].transpose(1, 2, 0) # (seq_len, fft_size, classes_num)

            if False:
                curr_audio_name = curr_meta['mixture_name']
                audio_path = os.path.join(audios_dir, curr_audio_name)
                (mixed_audio, fs) = read_audio(audio_path, target_fs=sample_rate, mono=True)
                
                out_wav_path = os.path.join(sep_wavs_dir, curr_audio_name)
                write_audio(out_wav_path, mixed_audio, sample_rate)
                
                window = np.hamming(window_size)
                mixed_stft_cmplx = stft(x=mixed_audio, window_size=window_size, hop_size=hop_size, window=window, mode='complex')
                mixed_stft_cmplx = mixed_stft_cmplx[0 : seq_len, :]
                mixed_stft = np.abs(mixed_stft_cmplx)
                
                for k in gt_indexes:
                    masked_stft = np.dot(pred_masks[:, :, k], inverse_melW) * mixed_stft
                    masked_stft_cmplx = real_to_complex(masked_stft, mixed_stft_cmplx)
                    
                    frames = istft(masked_stft_cmplx)
                    cola_constant = get_cola_constant(hop_size, window)
                    sep_audio = overlap_add(frames, hop_size, cola_constant)
                    
                    sep_wav_path = os.path.join(sep_wavs_dir, '{}_{}.wav'.format(os.path.splitext(curr_audio_name)[0], ix_to_lb[k]))
                    write_audio(sep_wav_path, sep_audio, sample_rate)
                    logging.info('Audio wrote to {}'.format(sep_wav_path))
      
            
        if True:
            for n in range(len(batch_output)):
            
                # if batch_audio_names[n] == '00013.wav':
                if True:
            
                    (events_stft, scene_stft, _) = generator.get_events_scene_mixture_stft(batch_audio_names[n])
                    
                    tmp = events_stft / (events_stft + scene_stft)
                    # tmp = add(events_stft, scene_stft)
                
                    
                    print(batch_audio_names[n])
                    print(batch_y[n])
                    target_labels = target_to_labels(batch_y[n], labels)
                    print(target_labels)
                    
                    fig, axs = plt.subplots(7, 7, figsize=(15, 10))
                    for k in range(classes_num):
                        axs[k // 6, k % 6].matshow(np.dot(batch_bottleneck[n, k], inverse_melW).T, origin='lower', aspect='auto', cmap='jet')
                        if labels[k] in target_labels:
                            color = 'r'
                        else:
                            color = 'k'
                        axs[k // 6, k % 6].set_title(labels[k], color=color)
                        axs[k // 6, k % 6].xaxis.set_ticks([])
                        axs[k // 6, k % 6].yaxis.set_ticks([])
                        axs[k // 6, k % 6].set_xlabel('time')
                        axs[k // 6, k % 6].set_ylabel('FFT bins')
                        
                    axs[6, 5].matshow(np.log(events_stft).T, origin='lower', aspect='auto', cmap='jet')
                    axs[6, 5].xaxis.set_ticks([0, 310])
                    axs[6, 5].xaxis.set_ticklabels(['0.0', '10.0 s'])
                    axs[6, 5].xaxis.tick_bottom()
                    axs[6, 5].yaxis.set_ticks([0, 1024])
                    axs[6, 5].yaxis.set_ticklabels(['0', '1025'])
                    axs[6, 5].set_xlabel('time')
                    axs[6, 5].set_ylabel('FFT bins')
                    
                    axs[6, 6].matshow(np.log(np.dot(events_stft, feature_extractor.melW)).T, origin='lower', aspect='auto', cmap='jet')
                    axs[6, 6].xaxis.set_ticks([0, 310])
                    axs[6, 6].xaxis.set_ticklabels(['0.0', '10.0 s'])
                    axs[6, 6].xaxis.tick_bottom()
                    axs[6, 6].yaxis.set_ticks([0, 63])
                    axs[6, 6].yaxis.set_ticklabels(['0', '64'])
                    axs[6, 6].set_xlabel('time')
                    axs[6, 6].set_ylabel('mel bins')
                        
                    # axs[6, 6].matshow(tmp.T, origin='lower', aspect='auto', cmap='jet')
                    # axs[6, 5].matshow(tmp.T, origin='lower', aspect='auto', cmap='jet')
                    # axs[6, 5].matshow(np.log(events_stft + scene_stft).T, origin='lower', aspect='auto', cmap='jet')
                        
                    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
                    # plt.tight_layout()
                    plt.show()
                    # return
                    
                    
def add2(args):
    
    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    scene_type = args.scene_type
    snr = args.snr
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size-overlap
    mel_bins = config.mel_bins
    seq_len = config.seq_len
    ix_to_lb = config.ix_to_lb
    
    thres = 0.1
    batch_size = 24
    
    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'scene_type={},snr={}'.format(scene_type, snr), 'development.h5')

    yaml_path = os.path.join(workspace, 'yaml_files', 'mixture.yaml')
    
    audios_dir = os.path.join(workspace, 'mixed_audios', 'scene_type={},snr={}'.format(scene_type, snr))
    
    '''
    t1 = time.time()
    with open(yaml_path, 'r') as f:
        data = yaml.load(f)
    print(time.time() - t1)
    
    import pickle
    create_folder(os.path.join(workspace, '_tmp'))
    pickle.dump(data, open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'wb'))
    '''
    
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    
    inverse_melW = feature_extractor.get_inverse_melW()
    
    
    meta = pickle.load(open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'rb'))

    # Data generator
    generator = InferenceDataGenerator(hdf5_path=hdf5_path,
                            batch_size=batch_size, 
                            holdout_fold=holdout_fold)

    generate_func = generator.generate_validate(data_type='validate', 
                                                    shuffle=False, 
                                                    max_iteration=None)
    
    
    # Evaluate on mini-batch
    for (iteration, data) in enumerate(generate_func):
        
        logging.info(iteration)
        
        (batch_x, batch_y, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        batch_gt_masks = []
        batch_mixture_stfts = []

        for n in range(len(batch_audio_names)):
            curr_meta = search_meta_by_mixture_name(meta, batch_audio_names[n])
            curr_events = curr_meta['events']
              
            gt_indexes = get_ground_truth_indexes(curr_events)
 
            gt_sed = get_sed_from_meta(curr_events) # (seq_len, classes_num)
              
            (events_stft, scene_stft, mixture_stft) = generator.get_events_scene_mixture_stft(batch_audio_names[n])
            # events_stft = np.dot(events_stft, feature_extractor.melW)
            # scene_stft = np.dot(scene_stft, feature_extractor.melW)
            
            gt_mask = ideal_ratio_mask(events_stft, scene_stft)    # (seq_len, fft_size)
            
            gt_masks = gt_mask[:, :, None] * gt_sed[:, None, :] # (seq_len, fft_size, classes_num)
            gt_masks = gt_masks.astype(np.float32)
            batch_gt_masks.append(gt_masks)
            
            batch_mixture_stfts.append(mixture_stft)
            
            # mixed_audio_path = os.path.join(audios_dir, batch_audio_names[n])
            # (mixed_audio, _) = read_audio(mixed_audio_path, target_fs=config.sample_rate, mono=True)
            # 
            # 
            # import crash
            # sdf
            
            
        if True:
            for n in range(len(batch_x)):
            
                # if batch_audio_names[n] == '00013.wav':
                if True:
            
                    print(batch_audio_names[n])
                    print(batch_y[n])
                    target_labels = target_to_labels(batch_y[n], labels)
                    print(target_labels)
                    
                    fig, axs = plt.subplots(7, 7, figsize=(15, 10))
                    for k in range(classes_num):
                        axs[k // 6, k % 6].matshow(batch_gt_masks[n][:, :, k].T, origin='lower', aspect='auto', cmap='jet')
                        if labels[k] in target_labels:
                            color = 'r'
                        else:
                            color = 'k'
                        axs[k // 6, k % 6].set_title(labels[k], color=color)
                        axs[k // 6, k % 6].xaxis.set_ticks([])
                        axs[k // 6, k % 6].yaxis.set_ticks([])
                        axs[k // 6, k % 6].set_xlabel('time')
                        axs[k // 6, k % 6].set_ylabel('FFT bins')
                        
                    # if False:
                    axs[6, 5].matshow(np.log(batch_mixture_stfts[n]).T, origin='lower', aspect='auto', cmap='jet')
                    axs[6, 5].xaxis.set_ticks([0, 310])
                    axs[6, 5].xaxis.set_ticklabels(['0.0', '10.0 s'])
                    axs[6, 5].xaxis.tick_bottom()
                    axs[6, 5].yaxis.set_ticks([0, 1024])
                    axs[6, 5].yaxis.set_ticklabels(['0', '1025'])
                    axs[6, 5].set_xlabel('time')
                    axs[6, 5].set_ylabel('FFT bins')
                        
                    # else:
                    axs[6, 6].matshow(np.log(np.dot(batch_mixture_stfts[n], feature_extractor.melW)).T, origin='lower', aspect='auto', cmap='jet')
                    axs[6, 6].xaxis.set_ticks([0, 310])
                    axs[6, 6].xaxis.set_ticklabels(['0.0', '10.0 s'])
                    axs[6, 6].xaxis.tick_bottom()
                    axs[6, 6].yaxis.set_ticks([0, 63])
                    axs[6, 6].yaxis.set_ticklabels(['0', '64'])
                    axs[6, 6].set_xlabel('time')
                    axs[6, 6].set_ylabel('mel bins')
                        
                    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
                    plt.show()
                    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_waveform = subparsers.add_parser('waveform')
    parser_waveform.add_argument('--workspace', type=str, required=True)
    parser_waveform.add_argument('--scene_type', type=str, required=True)
    parser_waveform.add_argument('--snr', type=int, required=True)
    parser_waveform.add_argument('--holdout_fold', type=int)
    parser_waveform.add_argument('--cuda', action='store_true', default=False)
    
    parser_mel_masks = subparsers.add_parser('mel_masks')
    parser_mel_masks.add_argument('--workspace', type=str, required=True)
    parser_mel_masks.add_argument('--model_type', type=str, required=True)
    parser_mel_masks.add_argument('--scene_type', type=str, required=True)
    parser_mel_masks.add_argument('--snr', type=int, required=True)
    parser_mel_masks.add_argument('--holdout_fold', type=int)
    parser_mel_masks.add_argument('--iteration', type=int, required=True)
    parser_mel_masks.add_argument('--cuda', action='store_true', default=False)
    
    parser_framework = subparsers.add_parser('framework')
    parser_framework.add_argument('--filename', type=str, required=True)
    parser_framework.add_argument('--workspace', type=str, required=True)
    parser_framework.add_argument('--scene_type', type=str, required=True)
    parser_framework.add_argument('--snr', type=int, required=True)
    parser_framework.add_argument('--holdout_fold', type=int)
    parser_framework.add_argument('--iteration', type=int, required=True)
    parser_framework.add_argument('--cuda', action='store_true', default=False)

    parser_a = subparsers.add_parser('a')
    parser_a.add_argument('--filename', type=str, required=True)
    parser_a.add_argument('--workspace', type=str, required=True)
    parser_a.add_argument('--scene_type', type=str, required=True)
    parser_a.add_argument('--snr', type=int, required=True)
    parser_a.add_argument('--holdout_fold', type=int)
    parser_a.add_argument('--iteration', type=int, required=True)
    parser_a.add_argument('--cuda', action='store_true', default=False)
    
    parser_b = subparsers.add_parser('b')
    parser_b.add_argument('--filename', type=str, required=True)
    parser_b.add_argument('--workspace', type=str, required=True)
    parser_b.add_argument('--scene_type', type=str, required=True)
    parser_b.add_argument('--snr', type=int, required=True)
    parser_b.add_argument('--holdout_fold', type=int)
    parser_b.add_argument('--iteration', type=int, required=True)
    parser_b.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.mode == 'waveform':
        plot_waveform(args)

    elif args.mode == 'mel_masks':
        plot_mel_masks(args)
        
    elif args.mode == 'framework':
        plot_framework(args)
    
    elif args.mode == 'a':
        add(args)
        
    elif args.mode == 'b':
        add2(args)
        
    else:
        raise Exception('Error!')