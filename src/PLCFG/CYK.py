# We can perform a syntax analysis of a string using the CYK algorithm.
class CYK:
    def __init__(self, _grammar):
        # We need a grammar to perform the analysis
        self.grammar = _grammar
        self.table = None
    
    # TODO: Implement a method which populates the probabilistic associated table
    # with the given string and the given grammar (Which is supossed to be in Chomsky Normal Form)