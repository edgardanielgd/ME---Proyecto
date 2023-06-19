import random
import numpy as np
from .Terminals import GrammarElement, SequenceElement

def subspans( sentence_length ):
     for length in range(2, sentence_length + 1):
          for i in range(1, sentence_length - length + 2):
               k = i + length - 1
               for j in range(i, k):
                    yield (i, j, k)

def ParseCYK( sentence, grammar ):
        # Array of booleans
        sentence_length = len(sentence)

        # Both P and T are indexed in the form [ RuleName, i, j], where i and j are the
        # indexes of the sentence, and RuleName is the name of the rule
        P = {}
        T = {}

        # Iterate over each terminal rule and initialize it with its probability


        #####
        # We will modify CYK a little bit for saving references to right brothers of each rule

        # Dict of dicts in the following way:
        # brothers[ rulename ] = {
        #   brother : count, # Count of times this brother appears
        #   brother2 : count2
        #   ...
        # }
        brothers = {} 

        
        # Terminals are related to a parent non terminal
        for i in range( len( sentence ) ):
            word = sentence[i]
            for name, non_terminal in grammar.non_terminals.items():
                for terminal_sequence in non_terminal.terminals:
                    # terminal_sequence is a SequenceElement object
                    if terminal_sequence.grammar_elements[0].terminal_value == word:
                        # This is the terminal to choose
                        P[ ( name, i, i ) ] = terminal_sequence.probability

                        # Construct a small tree where the root is the non terminal
                        # and the leaf is the terminal
                        # Each "Tree" will represent a rule in the grammar (which would be the root one)

                        # Now create a new terminal which represents this word as a leaf

                        word_terminal = GrammarElement( word, True )

                        # Now create the sequence where this terminal goes into
                        word_sequence = SequenceElement( [ word_terminal ] )

                        # Now create the non terminal which will be the root of the tree
                        parent_non_terminal = GrammarElement( name, False )

                        # And now create the tree
                        parent_non_terminal.add_sequence( word_sequence )
                        
                        # Finally add the non terminal to the tree
                        T[ ( name, i, i ) ] = parent_non_terminal
        
        # Now iterate over each non terminal rule and initialize it with its probability
        for i, j, k in subspans( sentence_length ):
            # We will iterate over the rules in the form X -> YZ [p]
            for name, non_terminal in grammar.non_terminals.items():
                # Iterate over each sequence in this non terminal

                for sequence in non_terminal.sequences:

                    if( len( sequence.grammar_elements ) == 2 ):
                        # Since we assume grammar is in CNF, there are at most 2 elements in each sequence
                        X = non_terminal
                        Y = sequence.grammar_elements[0]
                        Z = sequence.grammar_elements[1]
                        p = sequence.probability

                        # Z is the rightmost parent of Y, so save this
                        if not Y.name in brothers:
                            brothers[ Y.name ] = {}
                        
                        if not Z.name in brothers[ Y.name ]:
                            brothers[ Y.name ][ Z.name ] = 1
                        else:
                            brothers[ Y.name ][ Z.name ] += 1


                        # We will calculate the value of PYZ, but we should first
                        # Check if P[Y, i, j] and P[Z, j + 1, k] exist (if not, then the probability is 0)
                        if ( Y.name, i, j ) in P and ( Z.name, j + 1, k ) in P:
                            PYZ = P[ ( Y.name, i, j ) ] * P[ ( Z.name, j + 1, k ) ] * p
                        else:
                            PYZ = 0.0
                        
                        if (name, i, k) in P:
                            previous_P_X_i_k = P[ (name, i, k) ]
                        else:
                            previous_P_X_i_k = 0.0
                        
                        # Now check if P[X, i, k] is greater than PYZ
                        if PYZ > previous_P_X_i_k:
                            P[(name, i, k)] = PYZ

                            # Finally we will join T[Y, i, j], T[Z, j + 1, k] as children of T[X, i, k]

                            left = T[(Y.name, i, j)]
                            right = T[(Z.name, j + 1, k)]

                            # So first create the sequence
                            sequence = SequenceElement( [ left, right ] )

                            # Now create the non terminal
                            root_non_terminal = GrammarElement( name, False )

                            # And now create the tree
                            root_non_terminal.add_sequence( sequence )

                            # Finally add the non terminal to the tree
                            T[(name, i, k)] = root_non_terminal
        
        return T, brothers
                    
def get_path( CYKTree ):
    # Get the parent of the last word and return it
    # Later we will seek for another path in such way we can predict another
    # possible word
    path = [] # Save the path from the root to the rightmost leaf

    non_terminal = CYKTree

    path.append( non_terminal )

    while( not non_terminal.is_terminal ):
        # Get the first sequence (Actually there is only one)
        sequence = non_terminal.sequences[0]

        # Get the last element of the sequence
        last_element = sequence.grammar_elements[-1]

        # Get the non terminal of this element
        non_terminal = last_element

        path.append( non_terminal )
    
    
    return path
    
def predict_from_path( grammar, path, brothers, k ):
    # Path is a path from Non Terminals
    predicted = []

    # Get last terminal which generates the last word
    if len( path ) < 2:

        # Throw random words since path does not have enough elements
        values = list( grammar.terminals().keys() )
        probabilities = list( map(
            lambda x: x.probability,
            grammar.terminals().values()
        ))
        return np.random.choice(
            values, size = min( k, len( values ) ), p = probabilities, replace = False
        )
    
    # Calculate probabilities for brothers of last non terminal
    last_terminal = path[-2]

    brothers_values = list(brothers[ last_terminal.name ].keys())
    brother_probabilities = list(brothers[ last_terminal.name ].values())

    # Iterate until we have k predictions or we have no more brothers
    while len( brother_probabilities ) > 0:

        # Choose a random brother having weights
        u = np.random.uniform( 0, 1 )

        # We should normalize probabilities from brothers
        brothers_total = sum( brother_probabilities )

        for brother_id in range(len(brother_probabilities)):
            brother_probabilities[ brother_id ] = brother_probabilities[ brother_id ] / brothers_total
        
        # Now probabilities are normalized, so we can choose a random sequence
        # based on the probabilities
        brother_distribution = np.cumsum( brother_probabilities )

        for j in range( len( brother_distribution ) ):
            chosen_brother_probability = brother_probabilities[j]

            if u <= chosen_brother_probability:
                # Drop this brother from the list
                brother_probabilities.pop( j )
                chosen_brother = brothers_values.pop( j )
                chosen_brother = grammar.non_terminals[ chosen_brother ]

                # Choose a random terminal from this element

                terminals_values = chosen_brother.terminals
                terminal_probabilities = list( map(
                    lambda x: x.probability,
                    terminals_values
                ))

                while( len( terminal_probabilities ) > 0 ):
                    u = np.random.uniform( 0, 1 )

                    # We should normalize probabilities from terminals
                    terminal_probabilities_sum = sum( terminal_probabilities )

                    for terminal_id in range(len(terminal_probabilities)):
                        terminal_probabilities[ terminal_id ] = terminal_probabilities[ terminal_id ] / terminal_probabilities_sum

                    # Now probabilities are normalized, so we can choose a random sequence
                    # based on the probabilities

                    terminal_distribution = np.cumsum( terminal_probabilities )

                    for q in range( len( terminal_distribution ) ):
                        if u <= terminal_distribution[q]:
                            # This is the terminal to choose

                            # Drop this terminal from the list
                            terminal_probabilities.pop( q )
                            terminal_value = terminals_values.pop( q )

                            terminal = terminal_value.grammar_elements[0]

                            predicted.append( terminal.terminal_value )

                            if len( predicted ) >= k:
                                return predicted

                            break
                break
                
    return predicted