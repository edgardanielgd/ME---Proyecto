# But we have to store its ocurrences in order to calculate deeper probabilities
# Each element in sequence is a grammar element
class SequenceElement:
    def __init__(self, _grammar_elements):
        self.grammar_elements = _grammar_elements
        self.count = 0
        self.probability = 0.0
    
    # Overload of == operator
    def __eq__(self, elements_list ):
        # Custom comparator of sequences
        # Helpful for when we have to count each
        # sequence ocurrence
        if len( self.grammar_elements ) != len( elements_list ):
            return False
        
        for i in range( len( self.grammar_elements ) ):
            # This is a comparision between each pair of elements in array
            # Incoming array form: [ Name, Terminal ]
            # Terminal is a boolean value

            # Check if a given sequence ( as array of GrammarElement )
            # Matches another sequence

            given_item = elements_list[i]
            element = self.grammar_elements[i]
            if( element.is_terminal == given_item.is_terminal ):
                if( element.name != given_item.name ):
                    return False
            else:
                # Match terminal value
                if( element.terminal_value != given_item.terminal_value ):
                    return False

        return True
    
    # Calculate probability of this sequence globally
    def calculate_probability(self, parent_count):
        self.probability = self.count / parent_count

    def __str__(self):
        # Return a string representation of this sequence
        string_representation = ""
        for element in self.grammar_elements:
            string_representation += str(element.name) + " "
        
        # Append calculated probability
        string_representation += " - " + str(self.probability)
        return string_representation

# New GrammarElements (Terminals and Non Terminal tokens) will be created
# outside this class, but this class will be used to store the rules and relevant data
class GrammarElement:
        
    # Saves a set of rules
    def __init__(self, _value, is_terminal = False):
        self.name = None
        self.count = 0
        self.sequences = []
        self.terminals = []
        self.terminal_value = None 

        # Number of times this sequence ends a sentence
        self.ending_count = 0
        self.ending_prob = 0.0

        # Lets say rule can be whether a terminal or a non terminal
        if( is_terminal ):
            self.is_terminal = True
            self.terminal_value = _value
        else:
            self.is_terminal = False
            self.terminal_value = None
            self.name = _value
    
    def add_rule(self, _sequence ):
        # Add a rule to the rules array

        # Since this method will be called each time this rule is seen
        # Then we can increment the count of this rule
        self.count += 1
        
        # Search for the sequence
        for sequence in self.sequences:
            # Recall we defined a custom comparator for sequences
            if sequence == _sequence:
                # This sequence was found and then we should increment
                # its count rather than adding it to the array
                sequence.count += 1
                return
        
        # This sequence hasm't been seen yet
        # Then we should add it to the array and set its count
        new_sequence = SequenceElement( _sequence )
        new_sequence.count = 1

        # Check if sequence was a terminal
        if( len(_sequence) == 1 and _sequence[0].is_terminal ):
            self.terminals.append( new_sequence )
        else:
            self.sequences.append( new_sequence )

    def calculate_probabilities(self):
        # Calculate the probability of each sequence
        for sequence in self.sequences:
            sequence.calculate_probability( self.count )
        
        # Calculate the probability of each terminal
        for terminal in self.terminals:
            terminal.calculate_probability( self.count )
        
        # Calculate the probability of this rule ending a sentence
        self.ending_prob = self.ending_count / self.count

    def __str__(self):
        # String representation of this rule
        string_representation = ""
        for sequence in self.sequences:
            string_representation += f"""
                {self.name} -> {str(sequence)}
            """
        
        return string_representation