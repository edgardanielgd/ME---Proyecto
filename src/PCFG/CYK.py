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
        
        return T
                    
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
    
def predict_from_path( grammar, path, k ):
    # Path is a path from Non Terminals
    predicted = []
    for i in range( len( path ) - 1, -1, -1 ):
        gotten_non_terminal = path[i]
        
        # Actual non_terminal from original grammar
        non_terminal = grammar.non_terminals[ gotten_non_terminal.name ]

        if non_terminal is None:
            continue
            
        # Now we can take a different path which is another possible sequence of non terminals
        u = np.random.uniform( 0, 1 )

        # We should normalize probabilities from non terminals removing terminals probs
        # and then normalize them
        non_terminal_probabilities = []
        for sequence in non_terminal.sequences:
            non_terminal_probabilities.append( sequence.probability )
        
        # Normalize probabilities
        non_terminal_probabilities = np.array( non_terminal_probabilities )
        non_terminal_probabilities /= np.sum( non_terminal_probabilities )

        # Now probabilities are normalized, so we can choose a random sequence
        # based on the probabilities
        non_terminal_distribution = np.cumsum( non_terminal_probabilities )

        for j in range( len( non_terminal_distribution ) ):
            if u <= non_terminal_distribution[j]:
                # This is the sequence to choose
                elements = non_terminal.sequences[j].grammar_elements

                # Choose the last element of the sequence
                element = elements[-1]

                # Choose a random terminal from this element
                u = np.random.uniform( 0, 1 )

                # We should normalize probabilities from terminals
                terminal_probabilities = []
                for terminal in element.terminals:
                    terminal_probabilities.append( terminal.probability )

                # Normalize probabilities
                terminal_probabilities = np.array( terminal_probabilities )
                terminal_probabilities /= np.sum( terminal_probabilities )

                # Now probabilities are normalized, so we can choose a random sequence
                # based on the probabilities
                terminal_distribution = np.cumsum( terminal_probabilities )

                for q in range( len( terminal_distribution ) ):
                    if u <= terminal_distribution[q]:
                        # This is the terminal to choose
                        if len( predicted ) < k:
                            predicted.append( element.terminal_value )
            
    return predicted
                    
                
