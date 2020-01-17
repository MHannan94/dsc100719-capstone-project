import pandas as pd
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
import os
import pickle
from data_functions import show_sample_imgs
import matplotlib.pyplot as plt
%matplotlib inline



class ModelEvaluation():
    def __init__(self, model):
        self.model = model
        self.model.summary()
    
    def count_saved(self, directory= '.', condition= '.h5'):
        self.num_saved = len([file for file in os.listdir(directory) if condition in file])
    
    def train(self, x_train, y_train, epochs, batch_size, validation_size):
        self.model.compile(optimizer= 'adam',
                           loss= 'categorical_crossentropy',
                           metrics= ['accuracy'])
        self.hist = self.model.fit(x_train, y_train,
                                   epochs = epochs,
                                   batch_size= batch_size,
                                   validation_size= validation_size)
    
    def plot_loss(self):
        epochs = len(self.hist.history['loss'])
        loss_names = []
        for name in self.model.output_names:
            loss_names.append(name + '_loss')
            loss_names.append('val_' +name + '_loss')
        plt.style.use('ggplot')
        plt.figure(figsize= (12,15))
        plt.plot(np.arange(0, epochs), self.hist.history['loss'], label='train_loss')
        plt.plot(np.arange(0, epochs), self.hist.history[loss_names[0]], label='train_root_loss')
        plt.plot(np.arange(0, epochs), self.hist.history[loss_names[2]], label='train_vowel_loss')
        plt.plot(np.arange(0, epochs), self.hist.history[loss_names[4]], label='train_consonant_loss')
    
        plt.plot(np.arange(0, epochs), self.hist.history['val_loss'], label='val_train_loss', ls = '--')
        plt.plot(np.arange(0, epochs), self.hist.history[loss_names[1]], label='val_train_root_loss', ls = '--')
        plt.plot(np.arange(0, epochs), self.hist.history[loss_names[3]], label='val_train_vowel_loss', ls = '--')
        plt.plot(np.arange(0, epochs), self.hist.history[loss_names[5]], label='val_train_consonant_loss', ls = '--')
    
        plt.title('Loss against number of epochs')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()
    
    def plot_accuracy(self):
        epochs = len(self.hist.history['loss'])
        acc_names = []
        for name in self.model.output_names:
            acc_names.append(name + '_accuracy')
            acc_names.append('val_' +name + '_accuracy')
        plt.style.use('ggplot')
        plt.figure(figsize= (12,15))
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[0]], label='train_root_accuracy')
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[2]], label='train_vowel_accuracy')
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[4]], label='train_consonant_accuracy')
    
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[1]], label='val_train_root_accuracy', ls = '--')
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[3]], label='val_train_vowel_accuracy', ls = '--')
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[5]], label='val_train_consonant_accuracy', ls = '--')
    
        plt.title('Accuracy against number of epochs')
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        plt.show()
    
    def plot_metrics(self):
        self.plot_loss()
        self.plot_accuracy()
        
    def save_model(self):
        self.count_saved()
        self.model.save('model_{}.h5'.format(self.num_saved + 1))
        pickle.dump(self.hist, open('model_{}_history.p'.format(self.num_saved + 1), 'wb'))
        
    def test(self, x_test, y_test, sample_size= 10):
        idxs = show_sample_imgs(x_test, sample_size)
        root_probas = self.model.predict(x_test[idxs])[0]
        vowel_probas = self.model.predict(x_test[idxs])[1]
        consonant_probas = self.model.predict(x_test[idxs])[2]
        root_pred = np.argmax(root_probas, axis= 1)
        vowel_pred = np.argmax(vowel_probas, axis= 1)
        consonant_pred = np.argmax(consonant_probas, axis= 1)
        y_root = np.argmax(y_test[0][idxs], axis= 1)
        y_vowel = np.argmax(y_test[1][idxs], axis= 1)
        y_consonant = np.argmax(y_test[2][idxs], axis= 1)
        
    def evaluation(self, x_test, y_test):
        self.evaluation = self.model.evaluate(x_test, y_test)
        print(self.evaluation)