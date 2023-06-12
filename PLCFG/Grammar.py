import random
from Terminals import GrammarElement
from ChomskyNormalForm import ParseChomskyNormalForm
import nltk

# Save terminals and non_terminals to a specific class 
# for further processing
class Grammar:
    def __init__(self):
        self.non_terminals = {}
        self.terminals = {}

        # A start is a non terminal and will match the rule
        # starting a sentence
        self.non_terminals["START"] = GrammarElement(
            "START"
        )

        # In other side, END implies the end of a sentence
        # then non terminals targeting this terminal will
        # be the last rule in a generated sentence
        self.terminals["END"] = GrammarElement(
            "END", True
        )
    
    def learn_grammar_from_trees( self, sentences_trees ):
        # Print the first parsed sentence

        for parsed_sentence in sentences_trees:
            # Save each rule visited, the last one will be the end of the sentence
            # self.last_rule = None 

            # Parse the tree to Chomsky Normal Form before traversing it.
            # By doing this on all training sentences, we will have a PCFG in CNF as well
            ParseChomskyNormalForm( parsed_sentence )
            # parsed_sentence.chomsky_normal_form()

            # # First, we must define a START symbol which will be the root of the tree
            start_symbol = self.traverse(parsed_sentence)

            # # Add a rule to the start symbol, so we can check which
            # # terminal / non_terminal tends to start a sentence
            # self.non_terminals["START"].add_rule( [start_symbol] )

            # Increment by 1 the number of times last rule ends a sentence
            # self.last_rule.ending_count += 1
        
        # Calculate probabilities for each rule in the grammar
        for non_terminal, _ in self.non_terminals.items():
            self.non_terminals[non_terminal].calculate_probabilities()

    # Function for traversing a tree
    def traverse(self, tree):
        
        # Check if the tree is a leaf (Terminal)
        if type(tree) is str:
            if tree in self.terminals:
                # Increment realated GrammarElement count
                self.terminals[tree].count += 1
                return self.terminals[tree]
            else:
                # Terminal hasn't been seen yet
                new_terminal = GrammarElement(
                    tree, True
                )
                new_terminal.count = 1
                self.terminals[tree] = new_terminal
                return new_terminal
            
        # Check if the tree is a node
        if type(tree) is nltk.tree.Tree:
            # Get the label of the node
            label = tree.label()

            non_terminal_to_add = None

            if label not in self.non_terminals:
                # Non terminal hasn't been seen yet
                non_terminal_to_add = GrammarElement(
                    label, False
                )

                # Count will be updated after adding a new rule
                self.non_terminals[label] = non_terminal_to_add
            else:
                # Non terminal has been seen before
                non_terminal_to_add = self.non_terminals[label]

            # Now, visit each children 
            children = []

            for child in tree:
                # Traverse the child
                child_result = self.traverse(child)
                children.append( child_result )
            
            # Check if the last rule is the previous to a terminal
            # By overwritting this value, we will get at the end the
            # last Hidden Markov State that ends a sentence
            if len( children ) == 1 and children[0].is_terminal:
                self.last_rule = non_terminal_to_add

            # Add the child to the non terminal
            non_terminal_to_add.add_rule( children )    

            return non_terminal_to_add

        # This should never happen
        return None

    # Generate a sentence starting from a certain terminal
    def generate_sentence(self, non_terminal = None ):
        # Check if the starting terminal is in the grammar

        # Get the starting terminal
        if non_terminal is None:
            starting_non_terminal = self.non_terminals["S"]
        else:
            starting_non_terminal = self.non_terminals[non_terminal]
        
        # Generate a sentence starting from the starting terminal
        random_value = random.random()

        # We have to decide whether this non terminal
        # results in a sequence of non terminals or a single terminal

        # Choose a random terminal to generate
        for sequence in starting_non_terminal.terminals:
            if random_value <= sequence.probability:
                # This is the terminal to choose
                print( sequence.grammar_elements[0].terminal_value, end=" " )
                return

            # If not, then we must substract the probability of this terminal
            # and continue searching
            random_value -= sequence.probability
        
        # If not a terminal, then it must be a sequence of non terminals
        for sequence in starting_non_terminal.sequences:
            if random_value <= sequence.probability:
                # This is the sequence to choose
                for non_terminal in sequence.grammar_elements:
                    self.generate_sentence( non_terminal.name )
                return

            # If not, then we must substract the probability of this sequence
            # and continue searching
            random_value -= sequence.probability

    # Its required to a parse tree to be in Chomsky Normal Form
    # so the best solution is to take each sentence and convert it
    # Then our PCFG will be in CNF        