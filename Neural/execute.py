from Utils import *
from nltk.corpus import treebank
from Embedding import Embedding
from LSTM import lstm_cell_forward, lstm_forward, lstm_cell_backward, lstm_backward

# Slice to just some file ids (since words can exceed memory)
files = treebank.fileids()[:20]

# Get dictionaries for hot encodings
words = treebank.words( fileids = files )
word_to_index, index_to_word, vocabulary_size = build_hot_encodings( words )

# Get the size of the vocabulary
vocabulary_size = len( set( words ))
print( "Vocabulary size: ", vocabulary_size )

embedding_size = 10
distance = 2
load = True

# Create the embedding
embedding = Embedding( word_to_index, index_to_word, embedding_size, distance )

if load:
    print("Loading weights...")
    embedding.load_weights( "embedding.w" )
else:
    print("Training...")
    embedding.train( treebank.sents( fileids = files ), word_to_index )
    embedding.save_weights( "embedding.w" )

# We can now threat each word as a vector of size 10
# Now train a LSTM to predict the next word given a context
# Lets use the same distance as the embedding
lstm_size = 10

# TEST: Train a single sentence

# Initialize parameters
parameters = {
    "Wf": np.random.randn( lstm_size, lstm_size + embedding_size ) * 0.01,
    "bf": np.zeros( ( lstm_size, 1 ) ),
    "Wi": np.random.randn( lstm_size, lstm_size + embedding_size ) * 0.01,
    "bi": np.zeros( ( lstm_size, 1 ) ),
    "Wc": np.random.randn( lstm_size, lstm_size + embedding_size ) * 0.01,
    "bc": np.zeros( ( lstm_size, 1 ) ),
    "Wo": np.random.randn( lstm_size, lstm_size + embedding_size ) * 0.01,
    "bo": np.zeros( ( lstm_size, 1 ) ),
    "Wy": np.random.randn( vocabulary_size, lstm_size ) * 0.01,
    "by": np.zeros( ( vocabulary_size, 1 ) )
}

# Train a single sentence

# a_next, c_next, yt_pred, cache = lstm_forward(
#     np.array( [ word_to_index[ "the" ] ] ), 
#     # Initialize hidden states as zeros
#     np.zeros( ( lstm_size, 1 ) ),
#     parameters
# )


prediction = embedding.predict( "the", word_to_index, index_to_word )
print( get_top_predictions( prediction, index_to_word, 10 ) )
