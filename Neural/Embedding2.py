from .Dense import Dense
from .Utils import *
import numpy as np

class Embedding:
    def __init__(self, word_to_index, index_to_word, embedding_size = 10, distance = 2):
        self.embedding_size = embedding_size
        self.distance = distance
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

        # Get the number of words in the vocabulary
        self.vocabulary_size = len( word_to_index )

        # Build model
        self.input_dense = Dense( self.vocabulary_size, self.embedding_size, biasing = False )
        self.output_dense = Dense( self.embedding_size, self.vocabulary_size, biasing = False )
        self.input_dense.linkUpper( self.output_dense )

        self.training = None
    

    
    def create_training_from_sentences(self, sentences, only_nexts = False):
        # IMPORTANT: Index is an array, not a number

        self.training = []

        for sentence in sentences:
            # Iterate over each word in the sentence
            # Recall sentence is made up of arrays of word
            for i in range( len( sentence ) ):
                # Get the word and its context
                word = sentence[i]
                indexed_word = self.word_to_index[ word ]
                
                if( not only_nexts ):
                    # Generate for previous words
                    for j in range( max(0,i - self.distance), i ):

                        # Get the context word
                        context_word = sentence[j]
                        indexed_pair_word = self.word_to_index[ context_word ]

                        train_pair = ( indexed_word, indexed_pair_word)
                        self.training.append( train_pair )
                
                # Generate for next words
                for j in range( i + 1, min( len(sentence), i + self.distance + 1 ) ):

                    # Get the context word
                    context_word = sentence[j]
                    indexed_pair_word = self.word_to_index[ context_word ]

                    train_pair = ( indexed_word, indexed_pair_word )
                    self.training.append( train_pair )
                
        self.training = np.array( self.training ).astype( np.int16 )
        return self.training
    
    def train(self, sentences, epochs = 1000, learning_rate = 0.01):
        if self.training is None:
            self.create_training_from_sentences( sentences )
        
        training_length = len( self.training )

        print("Embedding training length: ", training_length )

        for epoch_id in range( epochs ):
            epoch_accuracy = 0
             
            # Shuffle data
            permutation = np.random.permutation( training_length )
            self.training = self.training[ permutation ]

            # Train
            for input_data, output_data in self.training:

                # Forward
                # Recall input_dense will forward the data to output_dense as well
                self.input_dense.forward( input_data )
                output_result = softmax( self.output_dense.output_data )

                # Backward
                error = output_result - output_data
                
                self.output_dense.backward( error, learning_rate )

                # Calculate accuracy
                if np.argmax( self.output_dense.output_data ) == np.argmax( output_data ):
                    epoch_accuracy += 1
            
            # Print epoch results
            epoch_accuracy /= training_length
            
            print( "Epoch: ", epoch_id, " Accuracy: ", epoch_accuracy )
    
    def save_weights(self, filename):
        self.input_dense.save_weights( filename + "_input" )
        self.output_dense.save_weights( filename + "_output" )
    
    def load_weights(self, filename):
        self.input_dense.load_weights( filename + "_input" )
        self.output_dense.load_weights( filename + "_output" )
    
    def predict(self, word, generate_indexed_word = False):
        if word not in self.word_to_index:
            # Choose a random word
            predicted_word = np.random.choice( list( self.word_to_index.keys() ) )

            if not generate_indexed_word:
                return self.word_to_index[ predicted_word ]
            
            return predicted_word
        
        # Get the index of the word
        indexed_word = self.word_to_index[ word ]

        # Forward
        self.input_dense.forward( indexed_word )

        # Get the output
        output = self.output_dense.output_data

        # Get the index of the output
        result = softmax( output )

        if not generate_indexed_word:
            return result

        # Get the word
        result = np.argmax( result )
        predicted_word = self.index_to_word[ result ]

        return predicted_word

    def get_embedding(self, word):
        # Returns the vector of weights on first layer associated with the word
        if word not in self.word_to_index:
            return None
        
        # Get the weights of the first layer
        weights = self.input_dense.weights # Shape: ( vocabulary_size, embedding_size )

        # Get the index of the word
        indexed_word = self.word_to_index[ word ]
        actual_index = get_index_from_hot_encoding( indexed_word )

        # Get the weights of the word
        word_weights = weights[ actual_index ]

        return word_weights