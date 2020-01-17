import pandas as pd
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt


def resize(df, size=64):
    resized = {}
    for i in range(df.shape[0]):
        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized

def create_training_data(df):
    X = df.drop(
            ['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic'],
            axis= 1)/255
    # CNN requires an input shape of (batch_size, h, w, channels)
    X = X.values.reshape(-1,64,64,1)
    # get dummies for target variables
    Y_root = pd.get_dummies(df['grapheme_root']).values
    Y_vowel = pd.get_dummies(df['vowel_diacritic']).values
    Y_consonant = pd.get_dummies(df['consonant_diacritic']).values
    return X, Y_root, Y_vowel, Y_consonant

def show_sample_imgs(imgs, sample_size):
    ncols = 5
    nrows = int(np.ceil(sample_size/ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize= (16,8))
    idxs = np.random.choice(len(imgs), sample_size, replace= False)
    for i,idx in enumerate(idxs):
        img = imgs[idx].reshape(64,64)
        ax.flatten()[i].imshow(img, cmap= 'gist_gray_r')
    plt.show()
    return idxs