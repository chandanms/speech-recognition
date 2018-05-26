# speech-recognition

Problem statement - Contains a few informational files and a folder of audio files. The audio folder contains subfolders with 1 second clips of voice commands, with the folder name being the label of the audio clip. There are more labels that should be predicted. The labels are yes, no, up, down, left, right, on, off, stop, go. Everything else should be considered either unknown or silence.

## My approach to the problem

I did quite a bit research on the topic and came up with the model that first converts the audio clip into a fixed size. The method used to do this is Mel-frequency cepstral coefficients (MFCCs). This vectors are treated as input to the Convolutional Neural Network (CNN). All the audios belonging to 10 labels are converted into numpy arrays and others are categorized as label 'unknown'. 
I coldn't find the audio clips for the label 'silent' so I excluded that from my training.  

## Getting started

Put the scripts, models and train.7z in the same directory. Extract train.7z and make sure the data is not corrupted. Install the required dependencies by,    

pip3 install -r requirements.txt

### Training

The training saves the model in the directory to use for inference and testing.  

python3 train.py  

My training results,  

![alt-text](https://raw.githubusercontent.com/chandanms/speech-recognition/master/Figure_1.png)


### Inference

To inference on a single audio clip using 
