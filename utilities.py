import os
import numpy as np
import soundfile
import librosa
import logging
import sed_eval

import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
    
    
def read_audio(path, target_fs=None, mono=True):
    
    (audio, fs) = soundfile.read(path)

    if mono:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
    if target_fs is not None and fs != target_fs:
        
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
            
        else:
            channels_num = audio.shape[1]
            
            audio = np.array(
                         [librosa.resample(audio[:, i], 
                                           orig_sr=fs, 
                                           target_sr=target_fs) 
                          for i in range(channels_num)]
                        ).T     # (samples_num, channels_num)
            
        fs = target_fs
        
    return audio, fs
    
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)
    
    
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na
    
    
def create_logging(log_dir, filemode):
    
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
    
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)
    
    
def calculate_average_energy(x):
    return np.mean(np.square(x))
    
    
# def normalize_to_unit_energy(x):
#     return x / np.sqrt(calculate_average_energy(x))
    
    
def normalize_to_energy(x, db):
    return x / np.sqrt(calculate_average_energy(x)) * np.power(10., db / 20.)
    
    
def calculate_scalar(x):

    if x.ndim <= 2:
        axis = 0
        
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std

def scale(x, mean, std):

    return (x - mean) / std
    
    
def sigmoid(x):
    
  return 1 / (1 + np.exp(-x))
  
  
def target_to_labels(y, labels):
    target_labels = []
    for k in range(len(y)):
        if y[k] == 1:
            target_labels.append(labels[k])
    return target_labels
    
    
def search_meta_by_mixture_name(meta, audio_name):
    
    for e in meta:
        if e['mixture_name'] == audio_name:
            return e
            
            
def get_ground_truth_indexes(events):
    
    lb_to_ix = config.lb_to_ix
    
    indexes = []
    
    for event in events:
        event_label = event['event_label']
        ix = lb_to_ix[event_label]
        indexes.append(ix)
        
    return np.array(indexes)
            
            
def get_sed_from_meta(events):
    
    classes_num = len(config.labels)
    seq_len = config.seq_len
    sample_rate = config.sample_rate
    overlap = config.overlap
    frames_per_sec = sample_rate / float(overlap)
    lb_to_ix = config.lb_to_ix
    
    gt_sed = np.zeros((seq_len, classes_num), dtype=np.int32)
    
    for event in events:
        event_label = event['event_label']
        onset = int(event['onset'] * frames_per_sec)
        offset = int(event['offset'] * frames_per_sec) + 1
        ix = lb_to_ix[event_label]
        gt_sed[onset : offset, ix] = 1
        
    return gt_sed
    
    
def get_sed_from_meta2(events, pred_indexes):
    
    classes_num = len(config.labels)
    seq_len = config.seq_len
    sample_rate = config.sample_rate
    overlap = config.overlap
    frames_per_sec = sample_rate / float(overlap)
    lb_to_ix = config.lb_to_ix
    
    gt_sed = np.zeros((seq_len, classes_num), dtype=np.int32)
    
    for event in events:
        event_label = event['event_label']
        onset = int(event['onset'] * frames_per_sec)
        offset = int(event['offset'] * frames_per_sec) + 1
        ix = lb_to_ix[event_label]
        gt_sed[onset : offset, ix] = 1
        
    gt_sed = gt_sed[:, pred_indexes]
        
    return gt_sed
    
    
def ideal_binary_mask(events_stft, scene_stft):
    db = 5.
    gt_mask = (np.sign(20 * np.log10(events_stft / (scene_stft + 1e-8) + 1e-8) + db) + 1.) / 2.
    return gt_mask
    
    
def tp_fn_fp_tn(y_gt, p_y_pred, thres, average):
    """
    Args:
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      tp, fn, fp, tn or list of tp, fn, fp, tn. 
    """
    if p_y_pred.ndim == 1:
        y_pred = np.zeros_like(p_y_pred)
        y_pred[np.where(p_y_pred > thres)] = 1.
        tp = np.sum(y_pred + y_gt > 1.5)
        fn = np.sum(y_gt - y_pred > 0.5)
        fp = np.sum(y_pred - y_gt > 0.5)
        tn = np.sum(y_pred + y_gt < 0.5)
        return tp, fn, fp, tn
    elif p_y_pred.ndim == 2:
        tps, fns, fps, tns = [], [], [], []
        n_classes = p_y_pred.shape[1]
        for j1 in range(n_classes):
            (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred[:, j1], y_gt[:, j1], thres, None)
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)
            tns.append(tn)
        if average is None:
            return tps, fns, fps, tns
        elif average == 'micro' or average == 'macro':
            return np.sum(tps), np.sum(fns), np.sum(fps), np.sum(tns)
        else: 
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")
        
        
def prec_recall_fvalue(y_gt, p_y_pred, thres, average):
    """
    Args:
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      prec, recall, fvalue | list or prec, recall, fvalue. 
    """
    eps = 1e-10
    if p_y_pred.ndim == 1:
        (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred, y_gt, thres, average=None)
        prec = tp / max(float(tp + fp), eps)
        recall = tp / max(float(tp + fn), eps)
        fvalue = 2 * (prec * recall) / max(float(prec + recall), eps)
        return prec, recall, fvalue
    elif p_y_pred.ndim == 2:
        n_classes = p_y_pred.shape[1]
        if average is None or average == 'macro':
            precs, recalls, fvalues = [], [], []
            for j1 in range(n_classes):
                (prec, recall, fvalue) = prec_recall_fvalue(p_y_pred[:, j1], y_gt[:, j1], thres, average=None)
                precs.append(prec)
                recalls.append(recall)
                fvalues.append(fvalue)
            if average is None:
                return precs, recalls, fvalues
            elif average == 'macro':
                return np.mean(precs), np.mean(recalls), np.mean(fvalues)
        elif average == 'micro':
            (prec, recall, fvalue) = prec_recall_fvalue(p_y_pred.flatten(), y_gt.flatten(), thres, average=None)
            return prec, recall, fvalue
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")
        
        
def prec_recall_fvalue_from_tp_fn_fp(tp, fn, fp, average):
    """
    Args:
      tp, fn, fp: int | list or ndarray of int
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      prec, recall, fvalue
    """
    eps = 1e-10
    if type(tp) == int or type(tp) == np.int32 or type(tp) == np.int64:
        prec = tp / max(float(tp + fp), eps)
        recall = tp / max(float(tp + fn), eps)
        fvalue = 2 * (prec * recall) / max(float(prec + recall), eps)
        return prec, recall, fvalue
    elif type(tp) == list or type(tp) == np.ndarray:
        n_classes = len(tp)
        if average is None or average == 'macro':
            precs, recalls, fvalues = [], [], []
            for j1 in range(n_classes):
                (prec, recall, fvalue) = prec_recall_fvalue_from_tp_fn_fp(tp[j1], fn[j1], fp[j1], average=None)
                precs.append(prec)
                recalls.append(recall)
                fvalues.append(fvalue)
            if average is None:
                return precs, recalls, fvalues
            elif average == 'macro':
                return np.mean(precs), np.mean(recalls), np.mean(fvalues)
        elif average == 'micro':
            (prec, recall, fvalue) = prec_recall_fvalue_from_tp_fn_fp(np.sum(tp), np.sum(fn), np.sum(fp), average=None)
            return prec, recall, fvalue
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect type!")
        
        
def event_based_evaluation(reference_event_list, estimated_event_list):
    """ Calculate sed_eval event based metric for challenge
        Parameters
        ----------
        reference_event_list : MetaDataContainer, list of referenced events
        estimated_event_list : MetaDataContainer, list of estimated events
        Return
        ------
        event_based_metric : EventBasedMetrics
        """

    files = {}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        # event_label_list=reference_event_list.unique_event_labels,
        event_label_list = config.labels, 
        t_collar=0.200,
        percentage_of_length=0.2,
    )

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        # events = []
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                # events.append(event.event_label)
        estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return event_based_metric