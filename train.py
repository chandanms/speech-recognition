import numpy as np
import librosa
import os
from keras.utils import to_categorical

DATA_PATH = './train/audio/'
def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


def save_data_to_array(path=DATA_PATH, max_pad_len=11):

	labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

	unknown_labels = ['bed', 'bird', 'cat', 'dog', 'eight', 'five', 'four', 'happy', 'house', 'marvin', 'nine', 'one', 'seven', 'sheila', 'six', 'three', 'tree', 'two', 'wow', 'zero']

	for label in labels:
		print ("Caculating the mfcc of " + str(label) + "...")
		mfcc_vectors = []
		wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
		for wavfile in wavfiles:
			mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
			mfcc_vectors.append(mfcc)
		np.save(label + '.npy', mfcc_vectors)

	unknown_mfcc_vectors = []
	for label in unknown_labels :
		print ("Calculating mfcc of unknown_labels...")		
		wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
		for wavfile in wavfiles :
			mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
			unknown_mfcc_vectors.append(mfcc)
	np.save('unknown' + '.npy', unknown_mfcc_vectors)

save_data_to_array() #Comment this out if the numpy arrays are already saved 

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


from sklearn.model_selection import train_test_split

def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown']

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)
    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


X_train, X_test, y_train, y_test = get_train_test()

X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

input_shape = (20, 11, 1)
nclasses = 11

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Activation, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(128, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.10))

model.add(Dense(nclasses, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train_hot, batch_size=100, epochs=50, verbose=1, validation_data=(X_test, y_test_hot))

model.save('saved_model.h5')
