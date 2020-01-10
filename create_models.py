import pandas as pd
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input

class ModelEvaluation():
    def __init__(self, model):
        self.model = model
        self.model.summary()
    
    def train(self, x, y, ):
        self.hist = self.model.fit(x, y,
                                   )