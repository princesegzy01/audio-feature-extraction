import numpy as np
import librosa
import librosa.display


import pandas as pd
import os
import csv
import sys


SR = 100
DURATION = 20


for folder in os.listdir("dataset/"):

    if(folder == ".DS_Store"):
        continue

    for filename in os.listdir("dataset/"+folder):

        if (filename[-3:] != "mp3"):
            continue

        y, sr = librosa.load("dataset/" + folder + "/" +
                             filename, duration=DURATION, sr=SR, mono=True)
        arr = list(y)
        arr.insert(0, filename)
        arr.insert(1, folder)

        file_time_series = open('time_series_dataset.csv', 'a', newline='')

        with file_time_series:
            writer = csv.writer(file_time_series)
            writer.writerow(arr)

        g = folder

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('dataset_feature.csv', 'a', newline='')
        with file:
            writer2 = csv.writer(file)
            writer2.writerow(to_append.split())

print("Done Feature extraction")
