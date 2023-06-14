import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class MainWindow( tk.Tk ):

    def __init__(self):
        super().__init__()
        self.title( "Next word prediction on rails!" )
        self.geometry( "600x400" )
        self.resizable( 0, 0 )
        self.create_widgets()
        self.configure(bg='blue')
        self.mainloop()
    
    def create_widgets(self):

        # Configure grid
        self.columnconfigure( 0, weight = 2 )
        self.columnconfigure( 1, weight = 1 )

        self.rowconfigure( 0, weight = 2 )
        self.rowconfigure( 1, weight = 1 )
        self.rowconfigure( 2, weight = 1 )

        # Label for a cool title
        self.title_label = ttk.Label( self, text = "Next word prediction on rails! :)" )
        self.title_label.grid(row=0,column=0, columnspan = 2, sticky = "nsew")

        # Frame for left column
        self.left_frame = ttk.Frame( self )
        self.left_frame.grid( row = 1, column = 0, sticky = "nsew" )

        # LEFT COLUMN

        # Top controls frame (Delete text, train and load)

        # Button for erasing
        self.erase_button = ttk.Button( self.left_frame, text = "Erase", command = self.erase )
        self.erase_button.pack( side = "top", fill = "x" )

        # Button for training
        self.train_button = ttk.Button( self.left_frame, text = "Train", command = self.train )
        self.train_button.pack( side = "top", fill = "x" )

        # Button for reading weights from previous train
        self.load_button = ttk.Button( self.left_frame, text = "Load Pre-Train", command = self.load )
        self.load_button.pack( side = "top", fill = "x" )

        # Text for input
        self.text = tk.Text( self.left_frame )
        self.text.insert( tk.END, "Write something here...")
        self.text.pack( side = "top", fill = "x" )

        # END LEFT COLUMN

        # Frame for right column
        self.right_frame = ttk.Frame( self )
        self.right_frame.grid( row = 1, column = 1, sticky = "nsew" )

        # RIGHT COLUMN

        # Label for method selection
        self.method_label = ttk.Label( self.right_frame, text = "Method:" )
        self.method_label.pack( side = "top", fill = "x" )

        # Combobox for method selection
        self.method_combobox = ttk.Combobox( self.right_frame, values = [ 
            "Word2Vec", "PCFG", "Ngram" 
        ])
        self.method_combobox.pack( side = "top", fill = "x" )

        # Label for next word predicted 
        self.next_word_label = ttk.Label( self.right_frame, text = "Next word:" )
        self.next_word_label.pack( side = "top", fill = "x" )

        # Table for next word predicted
        self.next_word_table = ttk.Treeview( self.right_frame, 
            columns = ( "word", "probability" ) 
        )
        self.next_word_table.heading( "#0", text = "Word" )
        self.next_word_table.heading( "word", text = "Word" )
        self.next_word_table.heading( "probability", text = "Probability" )
        self.next_word_table.pack( side = "top", fill = "both", expand = True )
    
    def erase(self):
        # Erase the text
        self.text.delete( "1.0", tk.END )
    
    def train(self):
        # Train the model
        messagebox.showinfo( "Training", "Training..." )
    
    def load(self):
        # Load the model
        messagebox.showinfo( "Loading", "Loading..." )
        
window = MainWindow()