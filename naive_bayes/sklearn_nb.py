from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import sys
easy = ['rec.autos','comp.os.ms-windows.misc','sci.space']
hard = ['comp.graphics','comp.sys.mac.hardware','comp.os.ms-windows.misc']
scope = easy
newsgroups_train = fetch_20newsgroups(subset='train',categories=scope)
newsgroups_test = fetch_20newsgroups(subset='test',categories=scope)

stop_words = set(["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost",
              "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",
              "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",
              "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand",
              "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but",
              "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail",
              "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere",
              "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few",
              "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found",
              "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he",
              "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself",
              "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it",
              "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may",
              "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must",
              "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody",
              "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one",
              "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own",
              "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
              "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so",
              "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system",
              "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there",
              "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third",
              "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too",
              "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very",
              "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where",
              "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
              "whither", "who", "whoever", "whole", "whom", "whose","why", "will", "with", "within", "without", "would",
              "yet", "you", "your", "yours", "yourself", "yourselves", "the", "i'm", "you're", "he's", "she's", "it's",
               "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd",
               "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't",
                "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't",
                 "can't", "cannot", "couldn't", "mustn't", "let's", "that's", "who's", "what's", "here's", "there's",
                  "when's", "where's", "why's", "how's", ".", ",", "-","i","say","says","said","kg","cm"])

# vectorize
#vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(stop_words = stop_words)
#vectorizer = CountVectorizer(stop_words = stop_words)
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

n =  vectors_train.shape[0]
label_percentage = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
lp = 0.01
for r in range(10):
  for lp in label_percentage:
    labels = np.random.permutation(n)[0:int(n/0.8*lp)]
    # train
    cnbg = GaussianNB()
    cnbg.fit(vectors_train.toarray()[labels],newsgroups_train.target[labels])
    pred = cnbg.predict(vectors_test.toarray())
    sys.stdout.write(str(metrics.precision_score(newsgroups_test.target, pred, average='weighted')) + '\t')
  sys.stdout.write('\n')