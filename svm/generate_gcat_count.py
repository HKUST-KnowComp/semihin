import numpy as np
import pickle
import sys
import random
import operator
from scipy.sparse import lil_matrix
# Data sets
path_20news = '../20news.hin'
path_gcat = '../GCAT_modified.hin'
path = path_20news
#path = path_gcat
# Scopes
hard = ['comp.graphics','comp.sys.mac.hardware','comp.os.ms-windows.misc']
easy = ['rec.autos','comp.os.ms-windows.misc','sci.space']
sim = ['GWEA', 'GDIS', 'GENV']
diff = ['GENT', 'GODD', 'GDEF']

scope = hard
scope_name = 'hard'

output_X = open(scope_name + 'X','w')
output_y = open(scope_name + 'y','w')
output_serials = open(scope_name + 'serials', 'w')
output_doc_serials = open(scope_name + 'doc_serials','w')

binary = True

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

stop_words = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost",
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
                  "when's", "where's", "why's", "how's", ".", ",", "-","i","say","says","said","kg","cm"]

def word_valid(w):
    if w in stop_words:
        return False
    if w.startswith('re'):
        return False
    if w.startswith('de'):
        return False
    if w.startswith('co'):
        return False
    if is_number(w):
        return False
    return True

def get_dwd_serials(scope):
    serials = {}
    doc_serials = {}
    f = open(path)
    flag = False
    for line in f.readlines():
        sp = line.split()

        # Document Header
        if line.startswith('doc'):
            docname = sp[0][4:]
            doctype = sp[2]
            # Constrain the data to be in 3 classes
            if doctype in scope:
                flag = True
                doc_serials[docname] = len(doc_serials)
            else:
                flag = False

        # The file in the line does not belong to class scopes
        elif flag == False:
            continue

        elif line.startswith('----sentence'):
            sp = line[20:].lower().split()
            for w in sp:
                if w not in serials and word_valid(w):
                    serials[w] = len(serials)

        # linkage from KB
        elif line.startswith('\t\t'):
            src = sp[0]
            dst = sp[2]
            if src not in serials:
                serials[src] = len(serials)
            if dst not in serials:
                serials[dst] = len(serials)

        # named entity
        elif line.startswith('\t'):
            key = sp[0]
            node_type = sp[1]
            if key not in serials:
                serials[key] = len(serials)

    f.close()
    return serials,doc_serials

def make_X(scope, serials, doc_serials):
    n = len(doc_serials)
    m = len(serials)
    print n,m
    X = lil_matrix((n, m))
    f = open(path)
    flag = False
    for line in f.readlines():
        sp = line.split()
        # Document Header
        if line.startswith('doc'):
            docname = sp[0][4:]
            doctype = sp[2]
            # Constrain the data to be in 3 classes
            if doctype in scope:
                flag = True
                docserial = serials[docname]
            else:
                flag = False

        elif flag == False:
            continue

        # word
        elif line.startswith('----sentence'):
            sp = line[20:].lower().split()
            for w in sp:
                if word_valid(w):
                    ws = serials[w]
                    # normalize dwd weight
                    if binary:
                    	X[docserial, ws] = 1
                    else:
                    	X[docserial, ws] = X[docserial, ws] + 1

        # named entity
        elif line.startswith('\t'):
            key = sp[0]
            nodeType = sp[1]
            keys = serials[key]
            if binary:
                X[docserial, keys] = 1
            else:
	            X[docserial, keys] = X[doc_serials, keys] + 1
	return X

def make_y(scope,doc_serials):
    f = open(path)
    label = np.zeros((len(doc_serials)))
    for line in f.readlines():
        sp = line.split()

        # Document Header
        if line.startswith('doc'):
            docname = sp[0][4:]
            doctype = sp[2]
            # Constrain the data to be in 3 classes
            if doctype in scope:
            	docserial = doc_serials[docname]
                label[docserial] = scope.index(doctype)
    f.close()
    return label

serials, doc_serials = get_dwd_serials(scope)
X = make_X(scope,serials,doc_serials)
y = make_y(scope,doc_serials)

pickle.dump(X,output_X)
pickle.dump(y,output_y)
pickle.dump(serials,output_serials)
pickle.dump(doc_serials,output_doc_serials)