import nltk
from nltk.corpus import gutenberg
import MarkovChain as mc


# Create a Markov Chain of order 1
mChain = mc.MarkovChain(1)

# Verify if the file exists
try:
    open("markov_chain.json")
    mChain.load("markov_chain")
except FileNotFoundError:
    nltk.download('gutenberg')

    texts = gutenberg.fileids()

    count = 0

    # Learn from the text
    for text in texts:
        mChain.learn_from_text(gutenberg.raw(text))
        count += 1
        print("Learned from " + text + " (" + str(count) + "/" + str(len(texts)) + ")")

    # Save the Markov Chain
    mChain.save("markov_chain")

# Predict the next word
print(mChain.next_word("I was walking down the", 5))
