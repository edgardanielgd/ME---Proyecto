from Utils import *
from nltk.corpus import treebank
from Embedding import Embedding
from LSTM import lstm_cell_forward, lstm_forward, lstm_cell_backward, lstm_backward

# Slice to just some file ids (since words can exceed memory)
files = treebank.fileids()[:100]

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
    embedding.load_weights( "embedding" )
else:
    print("Training...")
    embedding.train( treebank.sents( fileids = files ) )
    embedding.save_weights( "embedding" )

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



# Build input of shape (n_x, m, T_x)
# where n_x is the number of training examples
# m is the number of features (in this case the embedding size)
# T_x is the number of time steps


# First lets take the first 10 sentences and identify the maximum length
# of a sentence
n_sentences = 20

a0 = np.zeros( ( lstm_size, n_sentences ) )

sentences_slice = treebank.sents( fileids = files )[:n_sentences]
max_sentence_length = max( [ len( sentence ) for sentence in sentences_slice ] )

# Each word is a vector of size 10, and the maximum length is T_x
input_data = np.zeros( ( embedding_size, n_sentences, max_sentence_length ) )

for i, sentence in enumerate( sentences_slice ):
    for j, word in enumerate( sentence ):
        input_data[:, i, j] = embedding.get_embedding( word )
    
    # Not all sentences have the same length, so we need to pad the rest
    # with zeros
    for j in range( len( sentence ), max_sentence_length ):
        input_data[:, i, j] = np.zeros( ( embedding_size, 1 ) ).reshape( embedding_size )
    
print( "Input data shape: ", input_data.shape )

a, y, c, caches = lstm_forward(
    input_data, a0, parameters
)

# Calculate the loss
print( "y shape: ", y.shape )
loss = 0
for i in range( n_sentences ):
    for j in range( max_sentence_length ):
        loss += -np.log( y[ i, :, j ] )

print( "Loss: ", loss )

# Calculate loss for each word
loss = 0
for i in range( n_sentences ):  
    loss += -np.log( y[ i, :, max_sentence_length - 1 ] )

print( "Loss: ", loss )

# Calculate the gradients
da0, gradients = lstm_backward( y, caches )

# Calculate the gradients for each parameter
dWf = gradients[ "dWf" ]
dWi = gradients[ "dWi" ]
dWc = gradients[ "dWc" ]
dWo = gradients[ "dWo" ]
dWy = gradients[ "dWy" ]
dbf = gradients[ "dbf" ]
dbi = gradients[ "dbi" ]
dbc = gradients[ "dbc" ]
dbo = gradients[ "dbo" ]
dby = gradients[ "dby" ]

# Update the parameters
parameters[ "Wf" ] -= 0.01 * dWf
parameters[ "Wi" ] -= 0.01 * dWi
parameters[ "Wc" ] -= 0.01 * dWc
parameters[ "Wo" ] -= 0.01 * dWo
parameters[ "Wy" ] -= 0.01 * dWy
parameters[ "bf" ] -= 0.01 * dbf
parameters[ "bi" ] -= 0.01 * dbi
parameters[ "bc" ] -= 0.01 * dbc
parameters[ "bo" ] -= 0.01 * dbo
parameters[ "by" ] -= 0.01 * dby



# Train a single sentence

# a_next, c_next, yt_pred, cache = lstm_forward(
#     np.array( [ word_to_index[ "the" ] ] ), 
#     # Initialize hidden states as zeros
#     np.zeros( ( lstm_size, 1 ) ),
#     parameters
# )


# prediction = embedding.predict( "the", word_to_index, index_to_word )
# print( get_top_predictions( prediction, index_to_word, 10 ) )
