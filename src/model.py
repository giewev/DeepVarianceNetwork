import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

class BipolarClassifier(models.Model):

	# Initializes a new copy of the model
    def __init__(self, embedding_size, layer_sizes, signals = ['mean', 'variance']):
    	super(BipolarClassifier, self).__init__()
    	self.input_size = embedding_size * len(signals)
    	self.dense_layers = []
    	self.signals = signals
    	last_size = self.input_size
    	for layer_size in layer_sizes:
    		self.dense_layers.append(tf.keras.layers.Dense(layer_size, activation = 'tanh', input_shape = (last_size,)))
    		last_size = layer_size
    	# Layer for classification
    	self.dense_layers.append(tf.keras.layers.Dense(1, activation = 'tanh', input_shape = (last_size,)))

    # Takes a batch of user comments and classifies each of them as bipolar or not
    # If truth was provided, also calculates the loss
    def call(self, features, truth = None):
    	# signal = np.concatenate([features[x] for x in self.signals], axis = 1)
    	# signal = features['mean']
    	# print(signal.shape)
    	# print(signal)
    	signal = tf.concat([features[x] for x in self.signals], 1)

    	output_dict = dict()
    	output_dict[0] = signal
    	for ind,layer in enumerate(self.dense_layers):
    		signal = layer(signal)
    		output_dict[ind + 1] = signal
    	output_dict['outputs'] = signal

    	if truth is not None:
    		loss = tf.keras.losses.MSE(tf.convert_to_tensor(truth, dtype=tf.float32), signal)
    		# print(truth)
    		# print(signal.shape)
    		# loss = (truth - signal) ** 2
    		output_dict['loss'] = loss

    	# print(output_dict)
    	return output_dict['outputs'], output_dict['loss']