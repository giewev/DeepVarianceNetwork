import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

class MentalHealthClassifier(models.Model):

	# Initializes a new copy of the model
    def __init__(self, embedding_size, layer_sizes, num_classes, signals = ['mean', 'variance']):
    	super(MentalHealthClassifier, self).__init__()
    	self.input_size = embedding_size * len(signals)
    	self.dense_layers = []
    	self.signals = signals
    	self.num_classes = num_classes
    	last_size = self.input_size
    	for layer_size in layer_sizes:
    		self.dense_layers.append(tf.keras.layers.Dense(layer_size, activation = 'tanh', input_shape = (last_size,)))
    		last_size = layer_size
    	# Layer for classification
    	self.dense_layers.append(tf.keras.layers.Dense(num_classes, input_shape = (last_size,)))

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
    		# loss = tf.keras.losses.MSE(tf.convert_to_tensor(truth, dtype=tf.float32), signal)
    		loss = tf.nn.softmax_cross_entropy_with_logits(
			    labels = truth,
			    logits = output_dict['outputs'],
			)
    		output_dict['loss'] = loss

    	# print(output_dict)
    	return output_dict['outputs'], output_dict['loss']

class SoftmaxRegression(models.Model):

    # Initializes a new copy of the model
    def __init__(self, embedding_size, num_classes, signals = ['mean', 'variance']):
        super(SoftmaxRegression, self).__init__()
        self.input_size = embedding_size * len(signals)
        self.signals = signals
        self.num_classes = num_classes

        # Layer for classification
        self.dense_layer = tf.keras.layers.Dense(num_classes, input_shape = (self.input_size,))

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
        output_dict['outputs'] = self.dense_layer(signal)

        if truth is not None:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels = truth,
                logits = output_dict['outputs'],
            )
            output_dict['loss'] = loss

        return output_dict['outputs'], output_dict['loss']