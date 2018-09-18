import numpy as np
import h5py
import time
import logging

from utilities import calculate_scalar, scale
import config


class DataGenerator(object):

    def __init__(self, hdf5_path, batch_size, holdout_fold, seed=1234):
        """
        Inputs:
          hdf5_path: str
          batch_size: int
          holdout_fold: int
          seed: int, random seed
        """

        self.batch_size = batch_size
        self.holdout_fold = holdout_fold

        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()
        hf = h5py.File(hdf5_path, 'r')

        self.audio_names = np.array([s.decode() for s in hf['audio_name'][:]])
        self.x = hf['mixture_logmel'][:]
        self.y = hf['target'][:]
        self.folds = hf['fold'][:]

        hf.close()
        
        logging.info('Loading data time: {:.3f} s'.format(
            time.time() - load_time))

        # Split data to training and validation
        self.train_audio_indexes, self.validate_audio_indexes = \
            self.get_train_validate_audio_indexes()

        # Calculate scalar
        (self.mean, self.std) = calculate_scalar(
            self.x[self.train_audio_indexes])
            
    def get_train_validate_audio_indexes(self):
        
        audio_indexes = np.arange(len(self.audio_names))
        
        train_audio_indexes = audio_indexes[self.folds != self.holdout_fold]
        validate_audio_indexes = audio_indexes[self.folds == self.holdout_fold]
        
        return train_audio_indexes, validate_audio_indexes
    
    def generate_train(self):
        """Generate mini-batch data for training. 
        
        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size
        audio_indexes = np.array(self.train_audio_indexes)
        audios_num = len(audio_indexes)

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)
            batch_y = batch_y.astype(np.float32)

            yield batch_x, batch_y

    def generate_validate(self, data_type, shuffle, max_iteration=None):
        """Generate mini-batch data for evaluation. 
        
        Args:
          data_type: 'train' | 'validate'
          max_iteration: int, maximum iteration for validation
          shuffle: bool
          
        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
          batch_audio_names: (batch_size,)
        """

        batch_size = self.batch_size

        if data_type == 'train':
            audio_indexes = np.array(self.train_audio_indexes)

        elif data_type == 'validate':
            audio_indexes = np.array(self.validate_audio_indexes)

        else:
            raise Exception('Invalid data_type!')
            
        if shuffle:
            self.validate_random_state.shuffle(audio_indexes)

        audios_num = len(audio_indexes)

        iteration = 0
        pointer = 0

        while True:

            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[
                pointer: pointer + batch_size]
                
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]
            batch_audio_names = self.audio_names[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)
            batch_y = batch_y.astype(np.float32)

            yield batch_x, batch_y, batch_audio_names

    def transform(self, x):
        """Transform data. 
        
        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)
          
        Returns:
          Transformed data. 
        """

        return scale(x, self.mean, self.std)
        
    
class InferenceDataGenerator(DataGenerator):
    
    def __init__(self, hdf5_path, batch_size, holdout_fold):
        """Data generator for test data. 
        
        Inputs:
          dev_hdf5_path: str
          test_hdf5_path: str
          batch_size: int
        """
        
        super(InferenceDataGenerator, self).__init__(
            hdf5_path=hdf5_path, 
            batch_size=batch_size, 
            holdout_fold=holdout_fold)
            
        # Load stft data
        load_time = time.time()
        hf = h5py.File(hdf5_path, 'r')

        self.hf = hf
        
        logging.info('Loading data time: {:.3f} s'.format(
            time.time() - load_time))
        
    def generate_test(self):
        
        audios_num = len(self.test_x)
        audio_indexes = np.arange(audios_num)
        batch_size = self.batch_size
        
        pointer = 0
        
        while True:

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
                
            pointer += batch_size

            batch_x = self.test_x[batch_audio_indexes]
            batch_audio_names = self.test_audio_names[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_audio_names
            
    def get_events_scene_mixture_stft(self, audio_name):
        
        index = np.where(self.audio_names == audio_name)[0][0]
        
        events_stft = self.hf['events_stft'][index]
        scene_stft = self.hf['scene_stft'][index]
        mixture_stft = self.hf['mixture_stft'][index]
        
        return events_stft, scene_stft, mixture_stft