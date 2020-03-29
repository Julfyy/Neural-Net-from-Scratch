'''
This program is a simple neural network to classify binary numbers in range from 0 to 7

It has 3 neurons for input and 7 for output, and 5 hidden ones

Average accuracy is 99,79% on 20th epoch
Average cost value is 0.03

'''

from bin_classifier import Classifier
import numpy as np


#Dataset of binary numbers
dataset = [[0,0,0],
           [0,0,1],
           [0,1,0],
           [0,1,1],
           [1,0,0],
           [1,0,1],
           [1,1,0],
           [1,1,1]]

indices = np.random.randint(0,8, size=200)

#Create a dataset for training
x_train = [dataset[i] for i in indices]
y_train = [i for i in indices]


#Initialize a classifier object 
cl = Classifier(input_dim=3, output_dim=8, hidden_dim=5)

#Fit data
cl.fit_data(x_train,y_train, epochs=20)


#Make a single prediction
#cl.single_prediction([0,0,1])






