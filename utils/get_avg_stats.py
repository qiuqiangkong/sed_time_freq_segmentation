import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

from utilities import search_meta_by_mixture_name, event_based_evaluation
import config
from vad import activity_detection


def get_stats_single_fold(args):
    
    workspace = args.workspace
    filename = args.filename
    model_type = args.model_type
    scene_type = args.scene_type
    snr = args.snr
    
    labels = config.labels
    classes_num = len(labels)
    
    holdout_fold = 1
    
    # Paths
    stat_path = os.path.join(workspace, 'stats', filename, 
        'model_type={}'.format(model_type), 
        'scene_type={},snr={}'.format(scene_type, snr), 
        'holdout_fold{}'.format(holdout_fold), 'stat.p')
        
    pred_prob_path = os.path.join(workspace, 'pred_probs', filename, 
        'model_type={}'.format(model_type), 
        'scene_type={},snr={}'.format(scene_type, snr), 
        'holdout_fold{}'.format(holdout_fold), 'pred_prob.p')
    
    # Load stats
    stat = pickle.load(open(stat_path, 'rb'))
    
    pred_prob = pickle.load(open(pred_prob_path, 'rb'))
    
    meta = pickle.load(open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'rb'))
    
    audio_names = pred_prob['audio_name']
    sed_outputs = pred_prob['sed_output']
    sed_targets = pred_prob['sed_target']
    
    ref_event_list = get_ref_event_list(meta, audio_names)
    
    est_event_list = get_est_event_list(sed_outputs, audio_names, labels)
    
    event_based_metric = event_based_evaluation(ref_event_list, est_event_list)
    
    for key in stat.keys():
        print('{}: {:.4f}'.format(key, np.mean(stat[key])))
    
    print(event_based_metric)
    
    for n in range(len(audio_names)):
        
        curr_meta = search_meta_by_mixture_name(meta, audio_names[n])
        
        target_labels = [event['event_label'] for event in curr_meta['events']]
        
        print(target_labels)
        
        fig, axs = plt.subplots(7, 6, sharex=True, figsize=(15, 10))
        for k in range(classes_num):
            axs[k // 6, k % 6].plot(sed_outputs[n, :, k])
            axs[k // 6, k % 6].plot(sed_targets[n, :, k], color='r')
            axs[k // 6, k % 6].set_ylim(0, 1.02)
            if labels[k] in target_labels:
                color = 'r'
            else:
                color = 'k'
            axs[k // 6, k % 6].set_title(labels[k], color=color)
        
        plt.show()
        
    
def get_stats_all_folds(args):
    
    workspace = args.workspace
    filename = args.filename
    model_type = args.model_type
    scene_type = args.scene_type
    snr = args.snr
    
    labels = config.labels
    classes_num = len(labels)
    
    stat_list = []
    event_based_metric_list = []
    
    for holdout_fold in [1, 2, 3, 4]:
    
        # Paths
        stat_path = os.path.join(workspace, 'stats', filename, 
            'model_type={}'.format(model_type), 
            'scene_type={},snr={}'.format(scene_type, snr), 
            'holdout_fold{}'.format(holdout_fold), 'stat.p')
            
        pred_prob_path = os.path.join(workspace, 'pred_probs', filename, 
            'model_type={}'.format(model_type), 
            'scene_type={},snr={}'.format(scene_type, snr), 
            'holdout_fold{}'.format(holdout_fold), 'pred_prob.p')
        
        stat = pickle.load(open(stat_path, 'rb'))
        stat_list.append(stat)
        
        pred_prob = pickle.load(open(pred_prob_path, 'rb'))
        audio_names = pred_prob['audio_name']
        sed_outputs = pred_prob['sed_output']
        sed_targets = pred_prob['sed_target']
        
        meta = pickle.load(open(os.path.join(workspace, '_tmp', 'mixture_yaml.p'), 'rb'))
        
        ref_event_list = get_ref_event_list(meta, audio_names)
        
        est_event_list = get_est_event_list(sed_outputs, audio_names, labels)
        
        event_based_metric = event_based_evaluation(ref_event_list, est_event_list)
        event_based_metric_list.append(event_based_metric)
        
    avg_stat = average_stat_list(stat_list)
    
    avg_event_stat = average_event_based_metric_list(event_based_metric_list)
    
    for key in avg_stat.keys():
        print('{:<20}: {:.4f}'.format(key, np.mean(avg_stat[key])))

    for key in avg_event_stat.keys():
        print('{:<20}: {:.4f}'.format(key, np.mean(avg_event_stat[key])))
    
    print(avg_stat)
    print(avg_event_stat)
    
    
def get_ref_event_list(meta, audio_names):
    
    ref_event_list = []
    
    for (n, audio_name) in enumerate(audio_names):
        
        curr_meta = search_meta_by_mixture_name(meta, audio_name)
        
        events = curr_meta['events']
        
        for event in events:
            
            ref_event = {'filename': audio_name, 
                         'onset': event['onset'], 
                         'offset': event['offset'], 
                         'event_label': event['event_label']}
                         
            ref_event_list.append(ref_event)
    
    return ref_event_list
    
    
def get_est_event_list(sed_outputs, audio_names, labels):
    
    hop_size = config.window_size - config.overlap
    seconds_per_frame = hop_size / float(config.sample_rate)
    ix_to_lb = config.ix_to_lb

    est_event_list = []

    for (n, audio_name) in enumerate(audio_names):
        
        for k in range(len(labels)):
        
            bgn_fin_pairs = activity_detection(
                    sed_outputs[n, :, k], thres=0.2, 
                    low_thres=0.1, n_smooth=10, n_salt=10)
                    
            if len(bgn_fin_pairs) > 0:
                
                for [bgn, fin] in bgn_fin_pairs:
                
                    est_event = {'filename': audio_name, 
                            'onset': bgn * seconds_per_frame, 
                            'offset': fin * seconds_per_frame, 
                            'event_label': ix_to_lb[k]}
                    
                    est_event_list.append(est_event)
        
    return est_event_list


def average_stat_list(stat_list):
    
    keys = stat_list[0].keys()
    avg_stat = {}
    
    for key in keys:
        
        tmp = []
        for stat in stat_list:
            tmp.append(stat[key])
            
        avg_stat[key] = np.mean(tmp, axis=0)
        
    return avg_stat
    
    
def average_event_based_metric_list(event_based_metric_list):
    
    labels = config.labels
    event_stat = {}
    
    f_measures = []
    error_rates = []
    deletion_rates = []
    insertion_rates = []
    
    for label in labels:
    
        f_measure_list = []
        error_rate_list = []
        deletion_rate_list = []
        insertion_rate_list = []
    
        for event_based_metric in event_based_metric_list:
        
            tmp = event_based_metric.results_class_wise_metrics()[label]
            f_measure_list.append(tmp['f_measure']['f_measure'])
            error_rate_list.append(tmp['error_rate']['error_rate'])
            deletion_rate_list.append(tmp['error_rate']['deletion_rate'])
            insertion_rate_list.append(tmp['error_rate']['insertion_rate'])
            
        f_measures.append(np.mean(f_measure_list))
        error_rates.append(np.mean(error_rate_list))
        deletion_rates.append(np.mean(deletion_rate_list))
        insertion_rates.append(np.mean(insertion_rate_list))
        
    f_measures = np.array(f_measures)
    error_rates = np.array(error_rates)
    deletion_rates = np.array(deletion_rates)
    insertion_rates = np.array(insertion_rates)
    
    f_measures[np.isnan(f_measures)] = 0.
    error_rates[np.isnan(error_rates)] = 0.
    deletion_rates[np.isnan(deletion_rates)] = 0.
    insertion_rates[np.isnan(insertion_rates)] = 0.
        
    event_stat = {'f1_score': f_measures, 
                  'error_rate': error_rates, 
                  'deletion_rate': deletion_rates, 
                  'insertion_rate': insertion_rates}
                  
    return event_stat
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_single_fold = subparsers.add_parser('single_fold')
    parser_single_fold.add_argument('--workspace', type=str, required=True)
    parser_single_fold.add_argument('--filename', type=str, required=True)
    parser_single_fold.add_argument('--model_type', type=str, choices=['gmp', 'gap', 'gwrp'], required=True)
    parser_single_fold.add_argument('--scene_type', type=str, required=True)
    parser_single_fold.add_argument('--holdout_fold', type=int, required=True)
    parser_single_fold.add_argument('--snr', type=int, required=True)
    
    parser_all_folds = subparsers.add_parser('all_folds')
    parser_all_folds.add_argument('--workspace', type=str, required=True)
    parser_all_folds.add_argument('--filename', type=str, required=True)
    parser_all_folds.add_argument('--model_type', type=str, choices=['gmp', 'gap', 'gwrp'], required=True)
    parser_all_folds.add_argument('--scene_type', type=str, required=True)
    parser_all_folds.add_argument('--snr', type=int, required=True)
    
    args = parser.parse_args()

    if args.mode == 'single_fold':
        get_stats_single_fold(args) 
    
    elif args.mode == 'all_folds':
        get_stats_all_folds(args)
        
    else:
        raise Exception('Incorrect argument!')