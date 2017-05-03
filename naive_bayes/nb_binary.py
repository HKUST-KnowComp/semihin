import numpy as np
import pickle

# Data sets
path_20news = '../20news.hin'
path_gcat = '../GCAT_modified.hin'
#path = path_20news
path = path_gcat
# Scopes
hard = ['comp.graphics','comp.sys.mac.hardware','comp.os.ms-windows.misc']
easy = ['rec.autos','comp.os.ms-windows.misc','sci.space']
sim = ['GWEA', 'GDIS', 'GENV']
diff = ['GENT', 'GODD', 'GDEF']
scope = sim
scope_name = 'sim'

label_percentage = [0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60]
#lp = 0.60
# Naive Bayes parameters
entity = True
epsilon = 1e-3
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


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

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


def argmax(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def nb_train(label):
    prob_like = {}
    prob_prior = {}
    f = open(path)
    flag = False
    docName = ''
    docType = ''
    for line in f.readlines():
        if line.startswith('doc'):
            sp = line.split()
            docName = sp[0][4:]
            docType = sp[2]
            # If the data is in the training data
            if docType in scope and docName in label:
                prob_prior[docType] = prob_prior.get(docType, 0) + 1.0
                flag = True
            else:
                flag = False

        # If the data
        if flag == False:
            continue

        elif line.startswith('----sentence'):
            sp = line[20:].lower().split()
            for w in sp:
                if word_valid(w):
                    prob_like[docType + '_' + w] = prob_like.get(docType + '_' + w, 0) + 1.0
                    prob_like[docType] = prob_like.get(docType,0) + 1.0

        # link is not useful in naive Bayes
        elif line.startswith('\t\t'):
            pass

        # entities : just treat like bag of words
        elif line.startswith('\t') and entity:
            sp = line.split()
            key = sp[0]
            nodeType = sp[1]
            prob_like[docType + '_' + key] = prob_like.get(docType + '_' + key, 0) + 1.0
            prob_like[docType + '*'] = prob_like.get(docType) + 1.0

    f.close()
    return  prob_prior,prob_like

def nb_test(label,prob_prior,prob_like):
    correct = 0.0
    wrong = 0.0
    f = open(path)
    flag = False
    docName = ''
    docType = ''
    prob_posterior = {}
    for line in f.readlines():
        if line.startswith('doc'):
            if prob_posterior:
                if docType == argmax(prob_posterior):
                    correct += 1
                else:
                    wrong += 1
                prob_posterior = {}
            sp = line.split()
            docName = sp[0][4:]
            docType = sp[2]
            if docType in scope and docName in label:
                flag = True
            else:
                flag = False

        if flag == False:
            continue

        elif line.startswith('----sentence'):
            sp = line[20:].lower().split()
            for w in sp:
                if word_valid(w):
                    for c in scope:
                        prob_posterior[c] = prob_posterior.get(c,0.0) + np.log((prob_like.get(c + '_' + w, 0) + epsilon) / prob_like.get(c,0.0))

        # link is not useful in naive Bayes
        elif line.startswith('\t\t'):
            pass

        # entities : just treat like bag of words
        elif line.startswith('\t') and entity:
            sp = line.split()
            key = sp[0]
            nodeType = sp[1]
            for c in scope:
                prob_posterior[c] = prob_posterior.get(c,0.0) + np.log((prob_like.get(c + '_' + key, 0) + epsilon) / prob_like.get(c + '*',1.0))

    return correct / (correct + wrong)

def run():

    for lp in label_percentage:
        with open('../train/' + scope_name + '_' + '%.2f' % lp + '.lb') as f:
            train = pickle.load(f)
        with open('../test/' + scope_name + '_' + '%.2f' % lp + '.gt') as f:
            test = pickle.load(f)
        prob_prior, prob_like = nb_train(train)
        print '%.2f' % lp + ' %.4f' % nb_test(test,prob_prior,prob_like)

run()