import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import model_selection
from tensorflow import keras

csv_dir = "./ESC-50-master\\meta/esc50.csv"
audio_dir = "./ESC-50-master\\audio/"

data = pd.read_csv(csv_dir)

def load_audio_data(audio_dir, audio_file):
    file = os.path.join(audio_dir, audio_file)
    x, fs = librosa.load(file, sr=44100)
    return x, fs

def cal_mel(x):
    stft = np.abs(librosa.stft(x, n_fft=1024, hop_length=128)) ** 2
    log_stft = librosa.power_to_db(stft)
    mel = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return mel

def cal_mfcc(x):
    mfcc = librosa.feature.mfcc(x, n_mfcc=24, dct_type=3)
    return mfcc

x = list(data.loc[:, "filename"])
y = list(data.loc[:, "target"])

animals_file_name = []
animals_y = []
target_nums = [0, 2, 3, 5, 8]
class_num = len(target_nums)
for i in range(len(x)):
    idx = 0
    for j in target_nums:
        if y[i] == j:
            animals_file_name.append(x[i])
            animals_y.append(idx)
        idx += 1
        
X_train_file_name, X_test_file_name, y_train_pre, y_test_pre = model_selection.train_test_split(animals_file_name, animals_y, test_size=0.25, stratify=animals_y)

train_img_mlsp_dir = "./train_img_mlsp\\"
for i in range(len(X_train_file_name)):
    x, fs = load_audio_data(audio_dir, X_train_file_name[i])
    mel = cal_mel(x)
    librosa.display.specshow(mel, sr=fs)
    plt.savefig(train_img_mlsp_dir + X_train_file_name[i] + ".jpg", dpi=200)

test_img_mlsp_dir = "./test_img_mlsp\\"
for i in range(len(X_test_file_name)):
    x, fs = load_audio_data(audio_dir, X_test_file_name[i])
    mel = cal_mel(x)
    librosa.display.specshow(mel, sr=fs)
    plt.savefig(test_img_mlsp_dir + X_test_file_name[i] + ".jpg", dpi=200)
    
train_img_mfcc_dir = "./train_img_mfcc\\"
for i in range(len(X_train_file_name)):
    x, fs = load_audio_data(audio_dir, X_train_file_name[i])
    mfcc = cal_mfcc(x)
    librosa.display.specshow(mfcc, sr=fs)
    plt.savefig(train_img_mfcc_dir + X_train_file_name[i] + ".jpg", dpi=200)

test_img_mfcc_dir = "./test_img_mfcc\\"
for i in range(len(X_test_file_name)):
    x, fs = load_audio_data(audio_dir, X_test_file_name[i])
    mfcc = cal_mfcc(x)
    librosa.display.specshow(mfcc, sr=fs)
    plt.savefig(test_img_mfcc_dir + X_test_file_name[i] + ".jpg", dpi=200)