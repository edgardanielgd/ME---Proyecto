class NGramNLPModel:
    """
        Simple n-gram model for text prediction by training
        bayesian probabilities
    """

    # Default constructor
    def __init__(self, n=2):
        """
            Takes: n - the n-gram model to be used
            Recall n represents the number of previous words that will
            be considered as the past of a given word to appear again

            Example (Default):
                n = 2
        """
        self.n = n

        # Each word will be composed by a list of seen previous words
        # and the number of times that sequence did appear
        self.ngram = {}

        # Each word will be composed by a list of seen previous words
        # and the probability of that sequence to appear, this is the final
        # result after training model with data
        self.ngram_probabilities = {}
    
    def train_with_string (self, input_string):
        pass
