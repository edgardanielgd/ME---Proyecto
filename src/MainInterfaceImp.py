from MainInterface import Ui_MainWindow
from PySide6.QtWidgets import QMainWindow
import nltk
from Neural.Utils import *

class MainInterface(QMainWindow, Ui_MainWindow):
    def __init__(self, embeddingModel, predictorModel, ngramModel, markovModel, pcfgModel, predictedCount = 10 ):
        super(MainInterface, self).__init__()
        self.setupUi(self)

        # Add initial items to the combo box
        self.cmbModel.addItem("NGram")
        self.cmbModel.addItem("PCFG")
        self.cmbModel.addItem("Word Embedding")
        self.cmbModel.addItem("Neural Predictor")
        self.cmbModel.addItem("Markov Chain")

        # Handle combo box change
        self.cmbModel.currentIndexChanged.connect( self.predict )

        # Link reset text button
        self.btnErase.clicked.connect( self.resetText )

        # We will help user adding the word he needs to complete the sentence
        # when he clicks a certain item in the list
        self.lstPredictions.itemClicked.connect( self.completeSentence )

        self.embeddingModel = embeddingModel
        self.predictorModel = predictorModel
        self.ngramModel = ngramModel
        self.markovModel = markovModel
        self.pcfgModel = pcfgModel
        
        self.predictedCount = predictedCount

        self.txtText.textChanged.connect(self.checkUpdate)

        # We'll save current text state
        self.currentText = ""

        self.show()
    
    def checkUpdate(self):

        inputedText = self.txtText.toPlainText()

        if inputedText == self.currentText:
            return
        
        current_length = len(inputedText)

        # We will try to update as soon a new sentence structure is detected
        # We can do it simply by checking if the last character is a period, space, colon, etc.

        if current_length == 0:
            # Nothing to do
            self.currentText = inputedText
            return
        
        last_char = inputedText[-1]
        if last_char in [".", "!", "?", ":", ";", "\n", " ", ",", "(", ")", "[", "]", "{", "}"]:
            # We have a new sentence
            self.currentText = inputedText
            self.predict()
        
    def predict(self):
        # Get the current text
        text = self.txtText.toPlainText()

        # Get the current model
        model = self.cmbModel.currentText()

        # Get separated words
        words = nltk.word_tokenize(text)

        if len(words) == 0:
            return
        
        # Predict
        if model == "Word Embedding":

            # We are retraining our model with typed sentence
            # in order "to adapt" to the user's writing style
            # and make previous words more frequent
            # self.embeddingModel.special_train( words )
            
            lastWord = words[-1]
            
            rawPrediction = self.embeddingModel.predict( lastWord )
            predictions = get_top_predictions( rawPrediction, self.embeddingModel.index_to_word, self.predictedCount )
            self.updatePredictionsList( predictions )
        
        elif model == "Neural Predictor":
            
            rawPrediction = self.predictorModel.predict( words )
            predictions = get_top_predictions( 
                rawPrediction, self.embeddingModel.index_to_word, self.predictedCount 
            )
            self.updatePredictionsList( predictions )

        elif model == "NGram":

            predictions = self.ngramModel.generate_next_word( text, self.predictedCount )

            if predictions is None:
                return
            
            self.updatePredictionsList( predictions )

        elif model == "Markov Chain":
            predictions = self.markovModel.next_word( text, self.predictedCount )

            if predictions is None:
                return
            
            self.updatePredictionsList( predictions )
        
        elif model == "PCFG":
            predictions = self.pcfgModel.predict_next_word( text, self.predictedCount )

            if predictions is None:
                return
            
            self.updatePredictionsList( predictions )
            pass

    def resetText(self):
        self.txtText.clear()
        self.lstPredictions.clear()

        self.currentText = ""
    
    def updatePredictionsList( self, predictions ):
        # Clear the list
        self.lstPredictions.clear()
        
        for prediction in predictions:
            self.lstPredictions.addItem( prediction )

    def completeSentence(self, item):
        word = item.text()

        # Get the current text
        text = self.txtText.toPlainText()

        self.txtText.setText( text + word + " " )