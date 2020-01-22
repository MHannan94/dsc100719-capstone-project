import pandas as pd
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
import os
import pickle
from data_functions import show_sample_imgs
import matplotlib.pyplot as plt

class ModelEvaluation():
    def __init__(self, model, hist= []):
        self.model = model
        self.hist = hist
        self.model.summary()
    
    # to save files without overwriting existing saved models, count the number of saved models in the directory
    def count_saved(self, directory= '.', condition= '.h5'):
        self.num_saved = len([file for file in os.listdir(directory) if condition in file])
    
    # compile, train and save training history in a variable
    def train(self, x_train, y_train, epochs, batch_size, validation_split):
        self.model.compile(optimizer= 'adam',
                           loss= 'categorical_crossentropy',
                           metrics= ['accuracy'])
        self.hist = self.model.fit(x_train, y_train,
                                   epochs = epochs,
                                   batch_size= batch_size,
                                   validation_split= validation_split)
    
    # using the history, plot the loss and accuracy of the model against number of epochs
    def plot_loss(self):
        if not self.hist:
            return 'Model not trained. Run the train method on the class instance.'
        epochs = len(self.hist.history['loss'])
        # the names of the layers change everytime a new model is instantiated
        # create a list of the names of the target layers to access the loss
        # values from the history variable
        loss_names = []
        for name in self.model.output_names:
            loss_names.append(name + '_loss')
            loss_names.append('val_' +name + '_loss')
        plt.style.use('ggplot')
        plt.figure(figsize= (12,13))
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
        if not self.hist:
            return 'Model not trained. Run the train method on the class instance.'
        epochs = len(self.hist.history['loss'])
        # the names of the layers change everytime a new model is instantiated
        # create a list of the names of the target layers to access the accuracy
        # values from the history variable
        acc_names = []
        for name in self.model.output_names:
            acc_names.append(name + '_accuracy')
            acc_names.append('val_' +name + '_accuracy')
        plt.style.use('ggplot')
        plt.figure(figsize= (12,13))
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[0]], label='train_root_accuracy')
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[2]], label='train_vowel_accuracy')
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[4]], label='train_consonant_accuracy')
    
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[1]], label='val_train_root_accuracy', ls = '--')
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[3]], label='val_train_vowel_accuracy', ls = '--')
        plt.plot(np.arange(0, epochs), self.hist.history[acc_names[5]], label='val_train_consonant_accuracy', ls = '--')
    
        plt.title('Accuracy against number of epochs')
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()
    
    def plot_metrics(self):
        self.plot_loss()
        self.plot_accuracy()
        
    def save_model(self):
        self.count_saved()
        if not self.hist:
            return 'Model not trained. Run the train method on the class instance.'
        self.model.save('model_{}.h5'.format(self.num_saved + 1))
        pickle.dump(self.hist, open('model_{}_history.p'.format(self.num_saved + 1), 'wb'))
        
    def test(self, labels_df, x_test, y_test, sample_size= 10):
        # show images and get list of indexs for images
        idxs = show_sample_imgs(x_test, sample_size)
        # create dataframes with labels and components
        root_map = labels_df[
            labels_df['component_type'] == 'grapheme_root'].reset_index(drop=True)
        vowel_map = labels_df[
            labels_df['component_type'] == 'vowel_diacritic'].reset_index(drop=True)
        consonant_map = labels_df[
            labels_df['component_type'] == 'consonant_diacritic'].reset_index(drop=True)
        # predict labels and retrieve components from map dataframes
        root_probas = self.model.predict(x_test[idxs])[0]
        vowel_probas = self.model.predict(x_test[idxs])[1]
        consonant_probas = self.model.predict(x_test[idxs])[2]
        root_pred = np.argmax(root_probas, axis= 1)
        root_pred_comp = root_map.iloc[root_pred]['component'].values
        vowel_pred = np.argmax(vowel_probas, axis= 1)
        vowel_pred_comp = vowel_map.iloc[vowel_pred]['component'].values
        consonant_pred = np.argmax(consonant_probas, axis= 1)
        consonant_pred_comp = consonant_map.iloc[consonant_pred]['component'].values
        root = np.argmax(y_test[0][idxs], axis= 1)
        root_comp = root_map.iloc[root]['component'].values
        vowel = np.argmax(y_test[1][idxs], axis= 1)
        vowel_comp = vowel_map.iloc[vowel]['component'].values
        consonant = np.argmax(y_test[2][idxs], axis= 1)
        consonant_comp = consonant_map.iloc[consonant]['component'].values
        # create dataframe to display results
        col_level_names = [['predicted', 'true'],
                           ['root', 'r_comp',
                            'vowel', 'v_comp',
                            'consonant', 'c_comp']]
        cols= pd.MultiIndex.from_product(col_level_names)
        index= ['image_{}'.format(i) for i in range(len(root))]
        results = pd.DataFrame([root_pred, root_pred_comp,
                                vowel_pred, vowel_pred_comp,
                                consonant_pred, consonant_pred_comp,
                                root, root_comp, 
                                vowel, vowel_comp, 
                                consonant, consonant_comp],
                               index= cols,
                               columns= index).T
        return results
        
    def evaluation(self, x_test, y_test):
        self.evaluation = list(zip(self.model.metrics_names,
                                   self.model.evaluate(x_test, y_test))
                               )
        print(self.evaluation)