import nltk
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.book import *
import pickle

class NGramNLPModel:
    """
    Simple n-gram model for text prediction by training
    Bayesian probabilities
    """

    def __init__(self, n=2):
        """
        Takes: n - the n-gram model to be used
        Recall n represents the number of previous words that will
        be considered as the context for predicting the next word

        Example (Default):
            n = 2
        """
        self.n = n
        self.ngram = {}
        self.ngram_probabilities = {}

    def train_with_string(self, input_string):
        # Tokenize the input string into words
        words = nltk.word_tokenize(input_string)

        # Filter out punctuation symbols
        words = [word for word in words if word.isalnum()]

        # Generate n-grams from the tokenized words
        ngrams_list = list(ngrams(words, self.n + 1))

        # Count the occurrences of each n-gram
        for ngram in ngrams_list:
            prefix = tuple(ngram[:-1])
            suffix = ngram[-1]
            if prefix in self.ngram:
                if suffix in self.ngram[prefix]:
                    self.ngram[prefix][suffix] += 1
                else:
                    self.ngram[prefix][suffix] = 1
            else:
                self.ngram[prefix] = {suffix: 1}

        # Calculate probabilities for each n-gram
        for prefix in self.ngram:
            total_count = sum(self.ngram[prefix].values())
            self.ngram_probabilities[prefix] = {
                suffix: count / total_count
                for suffix, count in self.ngram[prefix].items()
            }

    def load_model(self, ngram, ngram_probabilities ):
        self.ngram = ngram
        self.ngram_probabilities = ngram_probabilities
    
    def generate_next_word(self, seed, predictionsCount = 1 ):

        if type(seed) == str:
            current_context = nltk.word_tokenize(seed)
        else:
            # For generating multiple texts, we will pass a list of words rather
            # than a string
            current_context = seed

        # Get initial prefix, could not belong to seen ngrams
        prefix = tuple(current_context[-self.n:])
        
        # Check if that context is in the model
        if prefix not in self.ngram_probabilities:
            # Search for a more similar context
            max_similarity = -1
            most_similar_context = None
            for context in self.ngram_probabilities:
                if context[:-1] == prefix[1:]:
                    # Using jaccard distance metric, note that we can consider using cosine distance as well
                    similarity = nltk.jaccard_distance(set(prefix), set(context))
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_context = context
            
            # If we found a similar context, we use it
            if most_similar_context is not None:
                prefix = most_similar_context
            else:
                # We couldn't find a similar context, so we return a random word
                next_words = np.random.choice(
                    list(map(
                        lambda x: x[-1],
                        self.ngram.keys()
                    )),
                    size = min(
                        predictionsCount, 
                        len(self.ngram.keys())
                    ),
                    replace = False
                )
                return next_words

        # If we reach this point, we have a valid prefix
        probabilities = self.ngram_probabilities[prefix]
        suffixes = list(probabilities.keys())
        probabilities = list(probabilities.values())
        next_words = np.random.choice(
            suffixes,
            size = min(predictionsCount, len(suffixes)),
            p = probabilities,
            replace = False
        )
        
        return next_words
  
    def generate_text(self, seed, length=10):
        current_context = nltk.word_tokenize(seed)
        generated_text = seed

        for _ in range(length):
            next_word = self.generate_next_word(current_context, 1)
            generated_text += " " + next_word[0]
            current_context.append(next_word[0])
            
        return generated_text

def create_ngram_model( load = False, n = 2):

    print( "===== Training NGRAM model" )

    model = NGramNLPModel(n=n)

    if not load:
        # Ejemplo de uso
        # nltk.download('punkt')

        # Entrenar con múltiples textos de NLTK
        input_data = ""
        input_data += nltk.corpus.gutenberg.raw('austen-emma.txt')
        input_data += nltk.corpus.gutenberg.raw('austen-persuasion.txt')
        input_data += nltk.corpus.gutenberg.raw('austen-sense.txt')
        input_data += nltk.corpus.gutenberg.raw('bible-kjv.txt')
        input_data += nltk.corpus.gutenberg.raw('blake-poems.txt')
        input_data += nltk.corpus.gutenberg.raw('bryant-stories.txt')
        input_data += nltk.corpus.gutenberg.raw('burgess-busterbrown.txt')
        input_data += nltk.corpus.gutenberg.raw('carroll-alice.txt')
        input_data += nltk.corpus.gutenberg.raw('chesterton-ball.txt')
        input_data += nltk.corpus.gutenberg.raw('edgeworth-parents.txt')


        # Train the model with 19 books
        model.train_with_string(input_data)
        books = [text1, text2, text3, text4, text5, text6, text7, text8, text9]
        for book in books:
            model.train_with_string(' '.join(book))

        # Save dictionaries to files
        with open("ngram.json", "wb") as outfile:
            pickle.dump(model.ngram, outfile)
        
        with open("ngram_probabilities.json", "wb") as outfile:
            pickle.dump(model.ngram_probabilities, outfile)
        
    else:

        # Load dictionaries from files
        with open("ngram.json", "rb") as infile:
            ngram = pickle.load(infile)
        
        with open("ngram_probabilities.json", "rb") as infile:
            ngram_probabilities = pickle.load(infile)

        model.load_model( ngram, ngram_probabilities )

    print( "===== Finishing Training NGRAM model" )

    return model
    
