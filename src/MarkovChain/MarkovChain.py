import json
import re
from string import punctuation

import numpy as np


class MarkovChain():
    # order: The order of the Markov Chain; how many words to consider
    # chain: The Markov Chain
    def __init__(self, order=1, chain={}):
        self.order = order
        self.chain = chain

    def clean_sentence(self, sentence):
        # Remove line breaks, tabs, etc.
        sentence = re.sub(r'[\n\r\t]', ' ', sentence)
        # Remove all leading and trailing spaces
        sentence = sentence.strip()
        # Convert to lowercase
        sentence = sentence.lower()

        return sentence

    def clean_words(self, words):
        # Remove all punctuation
        words = [w for w in words if w not in punctuation]

        # Remove punctuation from the start and end of words
        words = [w.strip(punctuation) for w in words]

        # Remove all numbers
        words = [w for w in words if w not in '0123456789']

        # Remove all empty words
        words = [w for w in words if w != '']

        # Remove all leading and trailing spaces
        words = [w.strip() for w in words]

        return words

    # Learn from a list of sentences

    def learn_from_text(self, text):
        # Split the text into sentences
        sentences = re.split(r'\s*[\.\?!;:\-\—\/][\'"\)\]]*\s*', text)
        for sentence in sentences:
            self.learn_from_sentence(sentence)

    # Learn from a single sentence
    def learn_from_sentence(self, sentence):
        # Clean the sentence
        sentence = self.clean_sentence(sentence)
        # Split the sentence into words
        words = sentence.split(' ')

        # Learn from the words
        self.learn_from_words(words)

    # Learn from a list of words (based on the order of the Markov Chain)
    def learn_from_words(self, words):
        # Clean the words
        words = self.clean_words(words)

        for index in range(len(words)-self.order-1):
            state = ' '.join(words[index:index + self.order])
            next_state = words[index + self.order + 1]

            if state not in self.chain:
                self.chain[state] = {}

            if next_state not in self.chain[state]:
                self.chain[state][next_state] = 1

            self.chain[state][next_state] += 1

    def next_word(self, text, top_n=1):
        # Get the last n (n = order) words
        sentence = re.split(r'\s*[\.\?!;:\-\—\/][\'"\)\]]*\s*', text)
        # Clean the sentence
        sentence = self.clean_sentence(sentence[-1])
        # Split the sentence into words
        words = sentence.split(' ')

        # Clean the words
        words = self.clean_words(words)

        # Get the state from the last n (n = order) words
        state = ' '.join(words[-self.order:])

        if state not in self.chain:
            return None

        words = list(self.chain[state].keys())

        weights = list(self.chain[state].values())

        sum_weights = sum(weights)

        probabilities = [weight / sum_weights for weight in weights]

        return np.random.choice(words, p=probabilities, size=min( top_n, len(words)), replace=False)

    # Save in JSON format
    def save(self, file_name):
        try:
            with open(file_name+'.json', "w") as file:
                json.dump({
                    "order": self.order,
                    "chain": self.chain
                }, file)
        except FileNotFoundError:
            print("Error: The file", file_name+'.json', "does not exist.")

    def load(self, file_name):
        try:
            with open(file_name+'.json', "r") as file:
                data = json.load(file)
                self.order = data["order"]
                self.chain = data["chain"]
        except FileNotFoundError:
            print("Error: The file", file_name+'.json', "does not exist.")
