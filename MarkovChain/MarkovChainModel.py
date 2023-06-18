import nltk
from nltk.corpus import gutenberg
from . import MarkovChain as mc

def create_markov_chain_model( load = False, order = 1 ):

    path = "MarkovChain/markov_chain"

    if load:
        # Create a Markov Chain of order 1
        mChain = mc.MarkovChain( order )

        # Verify if the file exists
        try:
            open( path + ".json")
            mChain.load( path )
        except FileNotFoundError:
            print("Error: The file markov_chain.json does not exist.")
            return None
    else:
        nltk.download('gutenberg')

        texts = gutenberg.fileids()

        count = 0

        # Learn from the text
        for text in texts:
            mChain.learn_from_text(gutenberg.raw(text))
            count += 1
            print("Learned from " + text + " (" + str(count) + "/" + str(len(texts)) + ")")

        # Save the Markov Chain
        mChain.save(path)

    return mChain
