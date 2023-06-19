from MainInterfaceImp import MainInterface
from Neural.NeuralPrediction import create_neural_model
from NGram.NGram import create_ngram_model
from MarkovChain.MarkovChainModel import create_markov_chain_model
from PCFG.GrammarDetect import create_grammar
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

    ###### MARKOV CHAIN MODEL #######
    ##############################

    markov = create_markov_chain_model( True, 2 )

    ###### END NGRAM MODEL #######
    ##############################

    ###### PCFG CHAIN MODEL #######
    ##############################

    pcfg = create_grammar( )

    ###### END NGRAM MODEL #######
    ##############################

    interface = MainInterface( embedding, predictor, ngram, markov, pcfg, 5 )
    sys.exit(app.exec())

if __name__ == "__main__":
    run()