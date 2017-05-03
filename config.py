import cPickle as pickle
import numpy as np
import os
# Permanent scope names
scope = dict(
    hard = ['comp.graphics','comp.sys.mac.hardware','comp.os.ms-windows.misc'],
    easy = ['rec.autos','comp.os.ms-windows.misc','sci.space'],
    full = ['comp.graphics', 'soc.religion.christian', 'comp.sys.mac.hardware', 'talk.politics.misc', 'rec.motorcycles', 'talk.religion.misc', 'comp.windows.x', 'comp.sys.ibm.pc.hardware', 'talk.politics.guns', 'alt.atheism', 'sci.med', 'sci.crypt', 'sci.space', 'misc.forsale', 'rec.sport.hockey', 'rec.sport.baseball', 'sci.electronics', 'comp.os.ms-windows.misc', 'rec.autos', 'talk.politics.mideast']
)

# Input File paths
data_path = '20news.hin'

# Repeats
repeats = 10

# word count to improve word feature
print os.getcwd()
f = open('data/word_count_20NG')
word_count = pickle.load(f)
f.close()

# Parameters
eps = 0.00001
alpha_candidates = [0.95, 0.9, 0.8]
label_percentage_candidate = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
tries = dict(
    scope=['easy'],
    alpha = np.linspace(0,0.99,100),
#    alpha=[0.98],
#    label_percentage=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    label_percentage=[0.01]
)
# Parameters for ensemble of typed linkage graph
tt1 = dict(
    scope = 'hard',
    alpha = 0.98,
    label_percentage = 0.01,
    types = ['computer','astronomy','organization','government','business','automotive','spaceflight','']
)

tt2 = dict(
    scope = 'easy',
    alpha = 0.9,
    label_percentage = 0.6,
    types = ['computer','astronomy','organization','government','business','automotive','spaceflight','']
)
tt3 = dict(
    scope = 'full',
    alpha = 0.98,
    label_percentage = 0.01,
    types = ['computer','astronomy','organization','government','business','automotive','']
)

typed_tries = tt2

# co-occurrence-link:
# link multiple entities that is in the same graph.
# we can adjust its weight and control it's level.
co_occurrence_link = False
co_occurrence_weight = 0.1
# - 's' : sentence level co-occurrence
# - 'd' : document level co-occurrence
co_occurrence_level = 's'

# weight for weighing the power of dwd and ded path
plain_dwd_weight = 0.112


word_freq_min_threshold = 3
word_freq_max_threshold = 100

# Output parameters
time_verbose = True
summary_verbose = True
summary_path = 'result/summary.txt'
result_verbose = True
result_path = 'galm/' + typed_tries['scope'] + '/labels.txt'

# Experiment types
# - 'plain' : all entity path are treated as equal.
# - 'dwd' : using entity path plus a dwd path.
# - 'typed' : type-constrained graph
graph_generator = 'dwd'

# a simple stop word list
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

