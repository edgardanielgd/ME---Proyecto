from .Dense import Dense
import numpy as np
from .Utils import softmax

# Create a model for predicting next word based only of joint embedding of previous words

class PredictionModel:
    def __init__(self, embedding, max_previous_words = 10 ):
        self.embedding = embedding

        # Build model
        self.max_previous_words = max_previous_words

        # Get vocabulary size
        self.vocabulary_size = embedding.vocabulary_size

        # We will always build a model 
        self.input_layer = Dense(
            # Concatenation of all previous words
            self.embedding.embedding_size * self.max_previous_words, 
            128, True
        )

        self.first_hidden_layer = Dense(
            128, 32, True
        )

        self.output_layer = Dense(
            32, self.vocabulary_size, True
        )

        # Link layers
        self.input_layer.linkUpper( self.first_hidden_layer )
        self.first_hidden_layer.linkUpper( self.output_layer )

        self.x_training = None
        self.y_training = None
    
    def train( self, sentences, epochs = 100, learning_rate = 0.01 ):
        # Create training data
        self.x_training = []
        self.y_training = []

        for sentence in sentences:
            for end_index in range( self.max_previous_words - 1, len(sentence) - 1 ):
                # Get the previous words
                previous_words = sentence[ end_index - self.max_previous_words + 1: end_index + 1 ]
                expected_word = sentence[ end_index + 1 ]

                # Generate input sequence
                input_sequence = np.array( [] )

                # Get the previous words embeddings
                for previous_word in previous_words:
                    # Get the embedding
                    previous_word_embedding = self.embedding.get_embedding( previous_word )

                    if previous_word_embedding is None:
                        # Add a zero vector
                        previous_word_embedding = np.zeros( self.embedding.embedding_size )

                    # Concatenate the embeddings
                    input_sequence = np.concatenate( ( input_sequence, previous_word_embedding ), axis = None )
                
                # Parse the expected word to its indexed representation
                expected_word_index = self.embedding.word_to_index[ expected_word ]

                # Add the training data
                self.x_training.append( input_sequence )
                self.y_training.append( expected_word_index )
        
        # Convert to numpy arrays
        self.x_training = np.array( self.x_training ).astype( np.float16 )
        self.y_training = np.array( self.y_training ).astype( np.int16 )

        training_length = len( self.x_training )

        print( "Predictor Training length: ", training_length )

        # Train the model
        for epoch_id in range( epochs ):
            epoch_accuracy = 0
             
            # Shuffle data
            permutation = np.random.permutation( training_length )
            self.x_training = self.x_training[ permutation ]
            self.y_training = self.y_training[ permutation ]

            # Train
            for sentence_id in range( training_length ):
                # Get the input and output data
                input_data = self.x_training[ sentence_id ]
                output_data = self.y_training[ sentence_id ]

                # Forward
                # Recall input_dense will forward the data to output_dense as well
                self.input_layer.forward( input_data )
                output_result = softmax( self.output_layer.output_data )

                # Backward
                error = output_result - output_data
                
                self.output_layer.backward( error, learning_rate )

                # Calculate accuracy
                if np.argmax( self.output_layer.output_data ) == np.argmax( output_data ):
                    epoch_accuracy += 1
            
            # Print epoch results
            epoch_accuracy /= training_length
            if epoch_id % 10 == 0:
                print( "Epoch: ", epoch_id, " Accuracy: ", epoch_accuracy )

    def save_weights( self, file_name ):
        self.input_layer.save_weights( file_name + "_input" )
        self.first_hidden_layer.save_weights( file_name + "_first_hidden" )
        # self.second_hidden_layer.save_weights( file_name + "_second_hidden" )
        self.output_layer.save_weights( file_name + "_output" )

    def load_weights( self, file_name ):
        self.input_layer.load_weights( file_name + "_input" )
        self.first_hidden_layer.load_weights( file_name + "_first_hidden" )
        # self.second_hidden_layer.load_weights( file_name + "_second_hidden" )
        self.output_layer.load_weights( file_name + "_output" )

    def predict( self, previous_words, generate_indexed_word = False ):
        # Generate input sequence
        input_sequence = np.array( [] )

        # SLice input to last max_previous_words
        previous_words = previous_words[ -min(len(previous_words), self.max_previous_words): ]

        # Get the previous words embeddings
        for previous_word in previous_words:
            # Get the embedding
            previous_word_embedding = self.embedding.get_embedding( previous_word )

            if previous_word_embedding is None:
                # Add a zero vector
                previous_word_embedding = np.zeros( self.embedding.embedding_size )

            # Concatenate the embeddings
            input_sequence = np.concatenate( ( input_sequence, previous_word_embedding ), axis = None )
        
        # Complete the input sequence with zeros
        input_sequence = np.concatenate( ( input_sequence, np.zeros( self.embedding.embedding_size * ( self.max_previous_words - len( previous_words ) ) ) ), axis = None )

        # Forward
        self.input_layer.forward( input_sequence )

        # Get the output
        output = softmax( self.output_layer.output_data )

        if not generate_indexed_word:
            return output

        # Get the word
        index = np.argmax( output )
        predicted_word = self.embedding.index_to_word[ index ]

        return predicted_word
