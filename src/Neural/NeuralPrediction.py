from .Embedding2 import Embedding
from .Model import PredictionModel
from nltk.corpus import treebank
import pickle

# Uncomment to download the treebank corpus
# import nltk
# nltk.download( 'treebank' )

from .Utils import *

def create_neural_model( 
    loadEmbedding = False, loadPredictor = False, embedding_size = 10, distance = 2,
    max_previous_words = 10, embedding_Epochs = 100, embedding_learning_rate = 0.01,
    predictor_Epochs = 100, predictor_learning_rate = 0.01
):
    path = "Weights/embedding"

    if not loadEmbedding:

        # Slice to just some file ids (since words can exceed memory)
        files = treebank.fileids()

        raw_sentences = treebank.sents( fileids = files )

        # Choose random k sentences
        k = 200

        # Randomize would prevent us from loading weights and make the training scalable
        # sentences = np.random.choice( raw_sentences, k, replace = False )
        sentences = raw_sentences[:k]

        sentences, words = clear_sentences( sentences )
        word_to_index, index_to_word, vocabulary_size = build_hot_encodings( words )

        # Get the size of the vocabulary
        vocabulary_size = len( set( words ))
        print( "Vocabulary size: ", vocabulary_size )

        # Save dicts and relevant data to files
        with open( "Weights/embedding_data.json", "wb" ) as file:
            pickle.dump( [ sentences, words, word_to_index, index_to_word, vocabulary_size ], file )
        
        # Create the embedding
        embedding = Embedding( word_to_index, index_to_word, embedding_size, distance )

        # TEST: Pass the same sentece multiple times to see if the accuracy
        embedding.train( sentences, embedding_Epochs, embedding_learning_rate )
        embedding.save_weights( path )

    else:
        with open( "Weights/embedding_data.json", "rb" ) as file:
            sentences, words, word_to_index, index_to_word, vocabulary_size = pickle.load( file )

        # Create the embedding
        embedding = Embedding( word_to_index, index_to_word, embedding_size, distance )

        print("Loading weights...")
        embedding.load_weights( path )

    max_previous_words = 5
    prediction_model = PredictionModel( embedding, max_previous_words )

    path = "Weights/prediction_model"

    if loadPredictor:
        print("Loading predictor weights...")
        prediction_model.load_weights( path )
    else:
        print("Training Predictor...")

        prediction_model.train( sentences, predictor_Epochs, predictor_learning_rate )
        prediction_model.save_weights( path )
    
    return embedding, prediction_model