### Sound Event Detection and Time-Frequency Segmentation from Weakly Labelled Data (PyTorch implementation)

Weakly labelled data refers to that only the presence or absence of sound events are known in an audio clip. The onset and offset annotations of sound events are unknown. We proposed a sound event detection and time-frequency segmentation framework trained only using the weakly labelled data. 

Figure here. 

## Dataset
The development dataset of DCASE 2018 Task 1 acoustic scene classification dataset is used as background noise. http://dcase.community/challenge2018/task-acoustic-scene-classification
DCASE 2018 Task 2 general audio tagging dataset is used as sound events: http://dcase.community/challenge2018/task-general-purpose-audio-tagging

|                   | Number of classes | Number of samples | Duration |
|-------------------|-------------------|-------------------|----------|
| DCASE 2018 Task 1 | 10                |                   |          |
| DCASE 2018 Task 2 | 41                |                   |          |

## Method
The proposed method is shown in the figure below. The red block is the training stage where only the weakly labelled data is used. The mapping *g*<sub>1</sub> is the segmentation mapping modeled by several convolutional layers. The mapping *g*<sub>2</sub> is the classification mapping modeled by a global pooling layer. The green and blue blocks indicate the sound event detection and sound event separation stages. 

## Run the code
**0. Prepare data**. The dataset looks like:

<pre>
.
└── TUT-urban-acoustic-scenes-2018-development
     ├── audio (8640 audios)
     │     └── ...
     ├── meta.csv
     └── ...   
└── DCASE 2018 Task 2
     ├── audio_train (9473 audios)
     │     └── ...
     ├── train.csv
     └── ...
</pre>

**1. Install dependent packages**. 

The code is implemented with Python 3 + PyTorch 0.4.0. You may need to install other independent packages with conda or pip. 

**2. Run**. 

Run the commands in runme.sh line by line, including:

(1) Modify the dataset path and workspace path. 

(2) Create yaml file containing the information of how the sound events and background noise are mixed. 

(3) Create mixed audio files. 

(4) Extract features. 

(5) Train network. 

(6) Inference. 

## Results


## References
[1] Kong, Qiuqiang, Yong Xu, Iwona Sobieraj, Wenwu Wang, and Mark D. Plumbley. "Sound Event Detection and Time-Frequency Segmentation from Weakly Labelled Data." arXiv preprint arXiv:1804.04715 (2018).
