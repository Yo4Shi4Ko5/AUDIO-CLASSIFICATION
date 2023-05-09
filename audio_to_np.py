import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
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

X_train_mlsp = []
y_train_mlsp = []
X_train_mfcc = []
y_train_mfcc = []
for i in range(len(X_train_file_name)):
    x, fs = load_audio_data(audio_dir, X_train_file_name[i])
    
    mel = cal_mel(x)
    mfcc = cal_mfcc(x)
    
    X_train_mlsp.append(mel)
    X_train_mfcc.append(mfcc)
    
    y_train_mlsp.append(y_train_pre[i])
    y_train_mfcc.append(y_train_pre[i])
X_train_mlsp = np.array(X_train_mlsp)
y_train_mlsp = np.array(y_train_mlsp)
y_train_mlsp = keras.utils.to_categorical(y_train_mlsp, class_num)
X_train_mfcc = np.array(X_train_mfcc)
y_train_mfcc = np.array(y_train_mfcc)
y_train_mfcc = keras.utils.to_categorical(y_train_mfcc, class_num)

X_test_mlsp = []
y_test_mlsp = []
X_test_mfcc = []
y_test_mfcc = []
for i in range(len(X_test_file_name)):
    x, fs = load_audio_data(audio_dir, X_test_file_name[i])
    
    mel = cal_mel(x)
    mfcc = cal_mfcc(x)
    
    X_test_mlsp.append(mel)
    X_test_mfcc.append(mfcc)
    
    y_test_mlsp.append(y_test_pre[i])
    y_test_mfcc.append(y_test_pre[i])
X_test_mlsp = np.array(X_test_mlsp)
y_test_mlsp = np.array(y_test_mlsp)
y_test_mlsp = keras.utils.to_categorical(y_test_mlsp, class_num)
X_test_mfcc = np.array(X_test_mfcc)
y_test_mfcc = np.array(y_test_mfcc)
y_test_mfcc = keras.utils.to_categorical(y_test_mfcc, class_num)  

X_train_mlsp = X_train_mlsp / abs(X_train_mlsp).max()
X_test_mlsp = X_test_mlsp / abs(X_test_mlsp).max()
X_train_mfcc = X_train_mfcc / abs(X_train_mfcc).max()
X_test_mfcc = X_test_mfcc / abs(X_test_mfcc).max()

np.savez("train_mlsp1.npz", x=X_train_mlsp, y=y_train_mlsp)
np.savez("test_mlsp1.npz", x=X_test_mlsp, y=y_test_mlsp)
np.savez("train_mfcc1.npz", x=X_train_mfcc, y=y_train_mfcc)
np.savez("test_mfcc1.npz", x=X_test_mfcc, y=y_test_mfcc)