import numpy as np
import librosa
import os

from keras.models import load_model

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown']

DATA_PATH = './train/audio'

def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

model = load_model('model_trained.h5')

lines = []

with open("./train/testing_list.txt", "r") as f :
	lines = f.readlines()

accuracy_rate = []

for line in lines :
	line = line.strip('\n')
	print ("Testing on " + str(line))
	label, wav = line.split('/')

	if label not in labels :
		label = 'unknown'
	wavmfcc = wav2mfcc(DATA_PATH + '/' + line)
	wavmfcc_reshaped = wavmfcc.reshape(1, 20, 11, 1)
	prediction = (labels[np.argmax(model.predict(wavmfcc_reshaped))])	

	if (prediction == label) :
		accuracy_rate.append(0)
	else :
		accuracy_rate.append(1)

print (accuracy_rate.count(0))
print (len(lines))
acc = accuracy_rate.count(0)/len(lines)

print ("Accuracy of the test is " + str(acc*100))



