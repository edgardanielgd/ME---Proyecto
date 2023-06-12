from nltk.corpus import treebank
from Grammar import Grammar
from CYKTest import ParseCYK
import numpy as np

# Dictionary for saving all of seen rules (Both leafs and nodes)
grammar = Grammar()

# Get parsed sentences from treebank corpus
parsed_sentences = treebank.parsed_sents()

# Get tagged sentences from treebank corpus
tagged_sentences = treebank.tagged_words()

# Traverse all parse trees and calculate probabilities 
# for each item on there
grammar.learn_grammar_from_trees( parsed_sentences )

# Get the maximum size of non terminals sequences
maximum = 0
for _, non_terminal in grammar.non_terminals.items():
    # non_terminal is a GrammarElement object
    for sequence in non_terminal.sequences:
        # sequence is a SequenceElement object
        if len(sequence.grammar_elements) > maximum:
            maximum = len(sequence.grammar_elements)


T = ParseCYK(
    ["mothers", "and", "daughters", "are", "talking", "to", "each", "other"],
    grammar
)

for tree in T.values():
    print( tree )
