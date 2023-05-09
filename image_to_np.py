import glob
from PIL import Image
import numpy as np
import keras
from keras.utils import np_utils

class_labels = ["0", "2", "3", "5", "8"]
class_num = len(class_labels)
train_img_mlsp_dir = "./train_img_mlsp\\"
test_img_mlsp_dir = "./test_img_mlsp\\"
train_img_mfcc_dir = "./train_img_mfcc\\"
test_img_mfcc_dir = "./test_img_mfcc\\"


X_train_mlsp = []
y_train_mlsp = []
idx = 0
for label in class_labels:
    files = glob.glob(train_img_mlsp_dir + "/*" + label + ".wav.jpg")
    for file in files:
        img = Image.open(file)
        img.convert("RGB")
        img = img.resize((128, 96))
        img = np.asarray(img)
        X_train_mlsp.append(img)
        y_train_mlsp.append(idx)
    idx += 1
X_train_mlsp = np.array(X_train_mlsp)
y_train_mlsp = np.array(y_train_mlsp)
X_train_mlsp = X_train_mlsp.astype("float32")
X_train_mlsp = X_train_mlsp / 255
y_train_mlsp = np_utils.to_categorical(y_train_mlsp, class_num)

X_test_mlsp = []
y_test_mlsp = []
idx = 0
for label in class_labels:
    files = glob.glob(test_img_mlsp_dir + "/*" + label + ".wav.jpg")
    for file in files:
        img = Image.open(file)
        img.convert("RGB")
        img = img.resize((128, 96))
        img = np.asarray(img)
        X_test_mlsp.append(img)
        y_test_mlsp.append(idx)
    idx += 1
X_test_mlsp = np.array(X_test_mlsp)
y_test_mlsp = np.array(y_test_mlsp)
X_test_mlsp = X_test_mlsp.astype("float32")
X_test_mlsp = X_test_mlsp / 255
y_test_mlsp = np_utils.to_categorical(y_test_mlsp, class_num)

X_train_mfcc = []
y_train_mfcc = []
idx = 0
for label in class_labels:
    files = glob.glob(train_img_mfcc_dir + "/*" + label + ".wav.jpg")
    for file in files:
        img = Image.open(file)
        img.convert("RGB")
        img = img.resize((128, 96))
        img = np.asarray(img)
        X_train_mfcc.append(img)
        y_train_mfcc.append(idx)
    idx += 1
X_train_mfcc = np.array(X_train_mfcc)
y_train_mfcc = np.array(y_train_mfcc)
X_train_mfcc = X_train_mfcc.astype("float32")
X_train_mfcc = X_train_mfcc / 255
y_train_mfcc = np_utils.to_categorical(y_train_mfcc, class_num)

X_test_mfcc = []
y_test_mfcc = []
idx = 0
for label in class_labels:
    files = glob.glob(test_img_mfcc_dir + "/*" + label + ".wav.jpg")
    for file in files:
        img = Image.open(file)
        img.convert("RGB")
        img = img.resize((128, 96))
        img = np.asarray(img)
        X_test_mfcc.append(img)
        y_test_mfcc.append(idx)
    idx += 1
X_test_mfcc = np.array(X_test_mfcc)
y_test_mfcc = np.array(y_test_mfcc)
X_test_mfcc = X_test_mlsp.astype("float32")
X_test_mfcc = X_test_mlsp / 255
y_test_mfcc = np_utils.to_categorical(y_test_mfcc, class_num)

np.savez("train_mlsp2.npz", x=X_train_mlsp, y=y_train_mlsp)
np.savez("test_mlsp2.npz", x=X_test_mlsp, y=y_test_mlsp)
np.savez("train_mfcc2.npz", x=X_train_mfcc, y=y_train_mfcc)
np.savez("test_mfcc2.npz", x=X_test_mfcc, y=y_test_mfcc)