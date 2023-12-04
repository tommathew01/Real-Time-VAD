import pyaudio
import numpy as np


import threading
import time
import python_speech_features # For exctracting features for deep learning
import librosa
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm # Progress meter
from sklearn import model_selection, preprocessing, metrics # Preparation data
from tensorflow.keras import models, layers
import tensorflow as tf
import pickle

CHUNK = 106200  # Chunk size for audio data
FORMAT = pyaudio.paInt16  # Input format
CHANNELS = 1  # Mono channel
RATE = 22050  # Sampling rate

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

model1 = models.load_model('vad_musan_32X32_model_equal.h5')

def findLabel(data, sr):
    # Set params for model:
    preemphasis_coef = 0.97 # Coefficient for pre-processing filter
    frame_length = 0.023 # Window length in sec
    frame_step = 0.01 # Length of step in sec
    num_nfft = 551 # Point for FFT
    num_features = 32 # Number of Mel filters
    n_frames = 32 # Number of frames for uniting in image
    frame_duration = 0.32
    features_logfbank = python_speech_features.base.logfbank(signal=data, samplerate=sr, winstep=frame_step, nfilt=num_features,
                                                                        nfft=num_nfft, lowfreq=0, highfreq=None, preemph=preemphasis_coef)

    num_frames = int(len(features_logfbank)/n_frames)

    # print(num_frames)

    single_audio_dataset  = list()
    for i in range(num_frames):
            spectrogram_image = features_logfbank[i*32:(i+1)*32]
            single_audio_dataset.append(spectrogram_image)

    
    X = np.array(single_audio_dataset)
    # print("shape is :   ",X.shape)
    # Reshaping for scaling:
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    # print("here:\n")
    # print(X.shape)

    # Scale data:
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    # print("here:\n")
    # print(X.shape)

    # And reshape back:
    X = X.reshape(X.shape[0], n_frames, n_frames)
    # print("here:\n")
    # print(X.shape)

    # Reshape data for convolution layer:
    stride = int(15)

    # X_train_reshaped = X[:int(np.floor(X.shape[0] / stride) * stride)]
    # print("here :\n")
    # print(X_train_reshaped.shape)

    X_train_reshaped = X.reshape(1, stride, n_frames, n_frames, 1)
    # print("here :\n")
    # print(X_train_reshaped.shape)



    output = model1.predict(X_train_reshaped)
    # print(output)

    output_after = (output>0.5).astype(int)
    y_pr = np.argmax(output_after, axis = 2)

    # print("here is :: ",y_pr)
    # # for i in range(len(y_pr)):
    # #   for j in range(15):
    # #         if(y_pr[i][j]==1):
    # #             y_pr[i][j]= 0
    # #         else:
    # #               y_pr[i][j] =1
                

    # print(y_pr)
    prediction = np.zeros(len(y_pr), dtype=int)
    for i in range(len(y_pr)):
        consecutive = 0;
        for j in range(15):
                if(y_pr[i][j]):
                    print("~", end="")
                    consecutive += 1
                    if(consecutive>=3):
                            prediction[i] = 1
                else:
                    consecutive = 0

    if(prediction[0] == 1):
        print("\nSOMEBODY IS TALKING")
    else:
        print("\nSILENCE")



import wave
data = []
try:
    print("Capturing audio... Press Ctrl+C to stop.")
    databuffer = [[],[]]
    i = 1
    k = 0
    
    while True:
        if(i == 0):
            i = 1
        else:
            i = 0
        chunk = stream.read(CHUNK)
        # audio_data = np.frombuffer(chunk, dtype=np.int16)
        # # Convert audio data to desired format (e.g., float32)
        # audio_data = audio_data.astype(np.float32) / 32767.0
        # findLabel(audio_data, RATE)
        # # data.append(audio_data)
        # print(audio_data)
        
        frames = []
        frames.append(chunk)
        wf = wave.open('sample_record1.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        data, sr = librosa.load('sample_record1.wav')
        findLabel(data, sr)



    # with open('my_list.pkl', 'wb') as file:
    #     pickle.dump(data, file)

except KeyboardInterrupt:
    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()