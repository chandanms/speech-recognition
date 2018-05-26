# speech-recognition

Problem statement - Contains a few informational files and a folder of audio files. The audio folder contains subfolders with 1 second clips of voice commands, with the folder name being the label of the audio clip. There are more labels that should be predicted. The labels are yes, no, up, down, left, right, on, off, stop, go. Everything else should be considered either unknown or silence.

## My approach to the problem

I did quite a bit research on the topic and came up with the model that first converts the audio clip into a fixed size. The method used to do this is Mel-frequency cepstral coefficients (MFCCs). This vectors are treated as input to the Convolutional Neural Network (CNN). All the audios belonging to 10 labels are converted into numpy arrays and others are categorized as label 'unknown'. 
I coldn't find the audio clips for the label 'silent' so I excluded that from my training.  

## Getting started

Put the scripts, models and train.7z in the same directory. Extract train.7z and make sure the data is not corrupted. Install the required dependencies by,    

pip3 install -r requirements.txt

### Training

The training saves the model in the directory to use for inference and testing. I have used all of the data to split it to train and validation. 

python3 train.py  

My training results after 50 epochs,    

![alt-text](https://raw.githubusercontent.com/chandanms/speech-recognition/master/Figure_1.png)
![alt-text](https://raw.githubusercontent.com/chandanms/speech-recognition/master/Figure_2.png)


### Inference

To inference on a single audio clip using pretrained model,

python3 inference.py -a path/to/audio/file

### Testing

The dataset has the list of audios to be tested in testing_list.txt. Currently it runs on trained_model.h5, if you want to run the test on your trained model, change the name of the model in code to the same. Run the test by,    

python3 test.py

I got an acuracy of 95% on testing.  

## Contacts

Email - mschandan96@gmail.com
Phone number - 9480118442



