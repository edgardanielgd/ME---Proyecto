from nltk.corpus import treebank
from Grammar import Grammar

# Dictionary for saving all of seen rules (Both leafs and nodes)
grammar = Grammar()

# Get parsed sentences from treebank corpus
parsed_sentences = treebank.parsed_sents()

# Get tagged sentences from treebank corpus
tagged_sentences = treebank.tagged_words()

# Traverse all parse trees and calculate probabilities 
# for each item on there
grammar.learn_grammar_from_trees( parsed_sentences )

print( len( parsed_sentences ) )