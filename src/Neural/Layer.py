class Layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def linkUpper( self, upper_layer ):
        self.upper_layer = upper_layer
        upper_layer.lower_layer = self
    
    def forward( self, input_data ):
        # Implemented in child class
        pass

    def backward( self, error, learning_rate = 0.01 ):
        # Implemented in child class
        pass
