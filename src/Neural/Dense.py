import numpy as np
from .Layer import Layer

class Dense(Layer):
    def __init__(self, input_dim, output_dim, biasing = True):
        # Input Dim is the number of neurons in the previous layer
        # Output Dim is the number of neurons in the current layer
        super().__init__( input_dim, output_dim )
        self.biasing = biasing

        # Initialize the weights and biases
        self.weights = np.random.randn( input_dim, output_dim ) * 0.01

        if biasing:
            self.biases = np.zeros( ( output_dim ) )
        
        self.upper_layer = None
        self.lower_layer = None
    
    def forward( self, input_data ):
        # Save the input data for backpropagation
        self.input_data = input_data

        # Calculate the output of the layer
        self.output_data = np.dot( input_data, self.weights )

        if self.biasing:
            self.output_data += self.biases

        if self.upper_layer is not None:
            self.upper_layer.forward( self.output_data )
        
        return self.output_data
    
    def backward( self, error, learning_rate = 0.01 ):
        # Error must have a length of output_dim
        self.w_gradients = np.dot( self.input_data[:, np.newaxis], error[np.newaxis, :] )

        # Calculate the error for the previous layer
        self.error = np.dot( error, self.weights.T )

        # Update the weights and biases
        self.weights -= self.w_gradients * learning_rate

        if self.biasing:
            # Update the biases
            self.b_gradients = error
            self.biases -= self.b_gradients * learning_rate

        if self.lower_layer is not None:
            self.lower_layer.backward( self.error, learning_rate )

        # Propagate the error to the next layer
        return self.error

    def save_weights( self, file_name ):
        np.save( file_name + "_weights", self.weights )

        if self.biasing:
            np.save( file_name + "_biases", self.biases )
    
    def load_weights( self, file_name ):
        self.weights = np.load( file_name + "_weights" + ".npy" )

        if self.biasing:
            self.biases = np.load( file_name + "_biases" + ".npy" )