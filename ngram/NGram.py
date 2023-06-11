import nltk
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.book import *

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

    def predict_next_word(self, prefix):
        if prefix in self.ngram_probabilities:
            probabilities = self.ngram_probabilities[prefix]
            suffixes = list(probabilities.keys())
            probabilities = list(probabilities.values())
            next_word = random.choices(suffixes, probabilities)[0]
            return next_word
        else:
            return None

    def generate_text(self, seed, length=10):
        current_context = nltk.word_tokenize(seed)
        generated_text = seed

        for _ in range(length):
            prefix = tuple(current_context[-self.n:])
            next_word = self.predict_next_word(prefix)
            if next_word:
                generated_text += " " + next_word
                current_context.append(next_word)
            else:
                max_similarity = -1
                most_similar_context = None
                for context in self.ngram_probabilities:
                    if context[:-1] == prefix[1:]:
                        similarity = nltk.jaccard_distance(set(prefix), set(context))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_context = context

                if most_similar_context:
                    probabilities = self.ngram_probabilities[most_similar_context]
                    suffixes = list(probabilities.keys())
                    probabilities = list(probabilities.values())
                    next_word = random.choices(suffixes, probabilities)[0]
                    generated_text += " " + next_word
                    current_context.append(next_word)
                else:
                    next_word = random.choice(list(self.ngram.keys()))[-1]
                    generated_text += " " + next_word
                    current_context.append(next_word)

        return generated_text





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

model = NGramNLPModel(n=2)
# Train the model with 19 books
model.train_with_string(input_data)
books = [text1, text2, text3, text4, text5, text6, text7, text8, text9]
for book in books:
    model.train_with_string(' '.join(book))


seed = "You are a fool"
print(model.generate_text(seed, length=50))
