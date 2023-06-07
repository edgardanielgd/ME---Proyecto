from Terminals import GrammarElement
import nltk

# Save terminals and non_terminals to a specific class 
# for further processing
class Grammar:
    def __init__(self):
        self.non_terminals = {}
        self.terminals = {}
    
    def learn_grammar_from_trees( self, sentences_trees ):
        # Print the first parsed sentence
        for parsed_sentence in sentences_trees:
            self.traverse(parsed_sentence)
        
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

            # Add the child to the non terminal
            non_terminal_to_add.add_rule( children )    

            return non_terminal_to_add

        # This should never happen
        return None