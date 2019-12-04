import tensorflow as tf
from tf.keras import models, layers

class BipolarClassifier(models.Model):

	# Initializes a new copy of the model
    def __init__(self, embedding_size, layer_sizes):
    	raise NotImplemented

    # Takes a batch of user comments and classifies each of them as bipolar or not
    # If truth was provided, also calculates the loss
    def call(self, inputs, truth = None):
    	raise NotImplemented