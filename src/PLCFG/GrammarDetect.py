from nltk.corpus import treebank
from Grammar import Grammar
from CYK import ParseCYK, get_rightmost_parent
import numpy as np

def create_grammar(load = False ):
    # Dictionary for saving all of seen rules (Both leafs and nodes)
    grammar = Grammar()

    if not load:
        # Get parsed sentences from treebank corpus
        parsed_sentences = treebank.parsed_sents()

        # Get tagged sentences from treebank corpus
        tagged_sentences = treebank.tagged_words()

        # Traverse all parse trees and calculate probabilities 
        # for each item on there
        grammar.learn_grammar_from_trees( parsed_sentences )
    else:
        pass
    pass

# Dictionary for saving all of seen rules (Both leafs and nodes)
grammar = Grammar()

# Get parsed sentences from treebank corpus
parsed_sentences = treebank.parsed_sents()

# Get tagged sentences from treebank corpus
tagged_sentences = treebank.tagged_words()

# Traverse all parse trees and calculate probabilities 
# for each item on there
grammar.learn_grammar_from_trees( parsed_sentences )


T = ParseCYK(
    ["mothers", "and", "daughters", "are", "talking", "to", "each", "other"],
    grammar
)

root = T[ ( "S", 1, 7 ) ]
path = get_rightmost_parent( root )
last_rule_name = path[-2].name
print( last_rule_name )

print( grammar.non_terminals[ last_rule_name ].terminals[0].grammar_elements[0].terminal_value )

# for tree in T.keys():
#     print( tree )
