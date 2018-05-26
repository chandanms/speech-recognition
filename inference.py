import numpy as np
import librosa
import os
import argparse

from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--audio", required=True, help="audio to detect")

args = vars(parser.parse_args())

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown']

DATA_PATH = './train/audio/'

def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

model = load_model('model_trained.h5')

sample = wav2mfcc(args["audio"])

sample_reshaped = sample.reshape(1, 20, 11, 1)

print(labels[np.argmax(model.predict(sample_reshaped))])
