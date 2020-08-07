import nltk
from nltk.corpus import brown
nltk.download('brown')
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
# print (brown.categories())

data = brown.sents(categories = ['adventure'])
# print (" ".join(data[8]))

#Tokenisation

document = """It was a very good movie. I really liked the cast and the plot was amazing. I went to the cinema hall to see it."""

sentence = "My name is Khushboo Gupta"

sents = sent_tokenize(document)
words = word_tokenize(sentence)  #breaks down special characters too
# print(words)

#Stop Word Removal

sw = set(stopwords.words('english'))
# print (sw)

text = "I am not a very good cricket player".split()
# print(text)

def remove_stopwords(text,stopwords):
    return [w for w in text if w not in stopwords]

# print(remove_stopwords(text,sw))

ps = PorterStemmer()
# print(ps.stem('running'))

corpus = ["If he had married her , he'd have been asking for trouble .","Sometimes he woke up in the middle of the night thinking of Ann , and then could not get back to sleep ."
          "His plans and dreams had revolved around her so much and for so long that now he felt as if he had nothing .",
          "The best antidote for the bitterness and disappointment that poisoned him was hard work ."]

cv = CountVectorizer()
vc = cv.fit_transform(corpus)

vc = vc.toarray()
# print(vc[0])
# print(vc[1])
# print(vc[2])


# print(len(vc))

def CustomTokenizer(document):
    words = word_tokenize(document.lower())
    words = remove_stopwords(words, sw)
    return words
# print(CustomTokenizer("This is a random text"))

cv = CountVectorizer(tokenizer= CustomTokenizer)
vc = cv.fit_transform(corpus).toarray()
print(vc)
# print(len(vc))
print(cv.vocabulary_)
