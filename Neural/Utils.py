import numpy as np

def build_hot_encodings( words ):
    words = list( set( words )) 
    total_length = len( words )

    # Results will be two dictionaries
    # One for the word to index
    # And another for the index to word
    word_to_index = {}
    index_to_word = {}

    # Iterate over each word
    for i in range( total_length ):

        word = words[i]
        if word in word_to_index:
            continue

        # Build array representation of the word
        hot_encodings = np.zeros( total_length )
        hot_encodings[ i ] = 1

        # Add the word to the dictionaries
        word_to_index[ word ] = hot_encodings

        # And its reverse
        index_to_word[ i ] = word
    
    return word_to_index, index_to_word, total_length

def softmax( arr, generate_indexed_word = False ):
    # Get the maximum value and create a zeros array
    # with the same size
    max_value = np.max( arr )

    # Subtract the maximum value to each element
    # to avoid overflow
    arr = arr - max_value

    # Calculate the exponential of each element
    exp_arr = np.exp( arr )

    # Calculate the sum of all exponential values
    sum_exp = np.sum( exp_arr )

    # Divide each exponential value by the sum
    # to get the softmax
    softmax_arr = exp_arr / sum_exp

    if not generate_indexed_word:   

        return softmax_arr
    else:
        # Now get the prediction
        prediction = np.argmax( softmax_arr )

        # And return word representation as array
        word_representation = np.zeros( len( arr ) )
        word_representation[ prediction ] = 1

        return word_representation

def get_index_from_hot_encoding( hot_encoding ):
    return np.argmax( hot_encoding )

def get_top_predictions( softmax_result, index_to_word, ntop = 5 ):
    # Get the top ntop predictions
    top_predictions = np.argsort( softmax_result )[-ntop:]

    # And return the words
    return [ index_to_word[ i ] for i in top_predictions ]

def clear_sentences( sentences ):
    # Remove punctuation and convert to lower case
    # Sentences is a matrix of words
    # Each row is a sentence
    # Each column is a word
    def check( word ):
        if word in ["!", ".", ",", "?", ":", ";", "\"", "\'"]:
            return False
        
        return True


    new_sentences = list(map(
        lambda sentence: list(filter(
            lambda word: check( word ),
            sentence
        )),
        sentences
    ))

    # Convert to lower case
    new_sentences = list(map(
        lambda sentence: list(map(       
            lambda word: word.lower(),
            sentence
        )),
        new_sentences
    ))

    # Get array of all words in sentences
    words = set([])
    for sentence in new_sentences:
        words = words.union( set( sentence ) )

    return new_sentences, words