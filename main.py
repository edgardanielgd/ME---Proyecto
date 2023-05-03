# Import required libraries
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import nltk

# The first time you run this code, you may see there is not any annotated data
# for tasks like pos tagging, so you need to download it first by uncommenting the following
# line:

# nltk.download()

# Firstly, get a comprhensive list of tags
# source of this list: https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk 

categories_classified = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition or subordinating conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'LS': 'List item marker',
    'MD': 'Modal',
    'NN': 'Noun, singular or mass',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund or present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb'
}


# Test string
input_string = '''
    Whats up Johan, how are you doing? I hope you are doing well. I am doing well too.
    Suffering from success? Like achieving 99% accuracy here but 0.0 as final calification anyways
    due to not having the social life like for exposing any of your hobbies as a proof of why
    you deserve a decent calification?'''

# Tokenizer model, will split words into arrays
detokenizer_model = TreebankWordDetokenizer()

# Inverse task than above
tokenizer_model = TreebankWordTokenizer()


# Split test string
tokens_for_input_string = tokenizer_model.tokenize(input_string)

# Get the POS tag for each token
pos_tags_for_input_string = nltk.pos_tag(tokens_for_input_string)

# Translate tags notation
for i in range(len(pos_tags_for_input_string)):

    # Word is the original word, tag is represented by means of a compressed notation
    word, tag = pos_tags_for_input_string[i]

    if tag in categories_classified:
        # Replace tag with its meaning
        pos_tags_for_input_string[i] = (
            word, categories_classified[tag]
            )
    
# Print tokens including their Part Of Speech (POS) tag
print(
    pos_tags_for_input_string
)
