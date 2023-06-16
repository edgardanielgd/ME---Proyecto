from MainInterfaceImp import MainInterface
from Neural.Embedding import create_embedding_model
from nltk.corpus import treebank
from Neural.Utils import *
from ngram.NGram import create_ngram_model
import sys
from PySide6.QtWidgets import QApplication

def run():
    app = QApplication(sys.argv)

    ###### EMBEDDING MODEL #######
    ##############################

    embedding, _, _, _ = create_embedding_model( True, 10, 2 )

    ###### END EMBEDDING MODEL #######
    ##############################

    ###### NGRAM MODEL #######
    ##############################

    ngram = create_ngram_model( True, 2 )

    ###### END NGRAM MODEL #######
    ##############################

    window = MainInterface( embedding, ngram, 5 )
    sys.exit(app.exec())

if __name__ == "__main__":
    run()