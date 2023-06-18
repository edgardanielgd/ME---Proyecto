from MainInterfaceImp import MainInterface
from Neural.NeuralPrediction import create_neural_model
from nltk.corpus import treebank
from Neural.Utils import *
from ngram.NGram import create_ngram_model
import sys
from PySide6.QtWidgets import QApplication

def run():
    app = QApplication(sys.argv)

    ###### EMBEDDING & NEURAL MODELS #######
    ##############################

    embedding, predictor = create_neural_model( True, True, 10, 2, 5, 1000, 0.01, 1000, 0.01 )

    ###### END EMBEDDING MODEL #######
    ##############################

    ###### NGRAM MODEL #######
    ##############################

    ngram = create_ngram_model( True, 2 )

    ###### END NGRAM MODEL #######
    ##############################

    interface = MainInterface( embedding, predictor, ngram, 5 )
    sys.exit(app.exec())

if __name__ == "__main__":
    run()