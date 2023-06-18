from Layer import Layer

class Relu(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) 
        out = x.copy()
        out[self.mask] = 0 

        if self.upper_layer is not None:
            self.upper_layer.forward( out )
        return out

    def backward(self, dout):
        dout[self.mask] = 0 
        dx = dout

        if self.lower_layer is not None:
            self.lower_layer.backward( dx )
            
        return dx