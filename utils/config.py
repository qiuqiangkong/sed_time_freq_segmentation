sample_rate = 32000     # Target sample rate during feature extraction
window_size = 2048      # Size of FFT window
overlap = 1024          # Amount of overlap between frames
clip_duration = 10.     # Duraion of an audio clip (seconds)
seq_len = 311   # Number of frames of an audio clip
mel_bins = 64   # Number of Mel bins

# kmax = 3

labels = ['Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 
          'Burping_or_eructation', 'Bus', 'Cello', 'Chime', 'Clarinet', 
          'Computer_keyboard', 'Cough', 'Cowbell', 'Double_bass', 
          'Drawer_open_or_close', 'Electric_piano', 'Fart', 'Finger_snapping', 
          'Fireworks', 'Flute', 'Glockenspiel', 'Gong', 'Gunshot_or_gunfire', 
          'Harmonica', 'Hi-hat', 'Keys_jangling', 'Knock', 'Laughter', 'Meow', 
          'Microwave_oven', 'Oboe', 'Saxophone', 'Scissors', 'Shatter', 
          'Snare_drum', 'Squeak', 'Tambourine', 'Tearing', 'Telephone', 
          'Trumpet', 'Violin_or_fiddle', 'Writing']

lb_to_ix = {lb: i for i, lb in enumerate(labels)}
ix_to_lb = {i: lb for i, lb in enumerate(labels)}