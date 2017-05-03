import csv
from hin import HIN
import misc

class DataReaderWang:
    ''' Data reader for Chenguang Wang's 20NG and RCV1 data

    '''

    def __init__(self, path='20news.hin',scope=[], data_set='20ng'):
        # Initial
        self.path = path
        self.scope = scope
        self.data_set = data_set

        # Attributes Declaration
        self.Ids = {}
        self.DocIds = {}
        self.DocLabels = {}
        self.Links = []
        self.RelationIds = {}
        self.NodeTypes = {}
        self.TypeIds = {}
        self.WordIds = {}
        self.WordDFs = {}
        self.WordCount = {}
        self.EntityDFs = {}

        # Default Values
        self.TypeIds['document'] = 0
        self.TypeIds['word'] = 1
        self.RelationIds['document_word'] = 0
        self.RelationIds['_document_word'] = 1

    def readfile(self):
        if self.data_set == '20ng':
            from misc import word_valid
        else:
            from misc_gcat import word_valid
        f = open(self.path)
        self.Ids = {}
        self.DocIds = {}

        scope_flag = False
        for line in f.readlines():
            sp = line.split()

            # Document Header
            if line.startswith('doc'):
                doc_name = sp[0][4:]
                doc_type = sp[2]
                # Constrain the data to be in 3 classes
                if doc_type in self.scope:
                    scope_flag = True
                    doc_id = len(self.Ids)
                    self.Ids[doc_name] = doc_id
                    self.NodeTypes[doc_id] = self.TypeIds['document']
                    self.DocIds[doc_name] = doc_id 
                    self.DocLabels[doc_id] = doc_type
                else:
                    scope_flag = False

            # Skip the line that is out the scope
            elif not scope_flag:
                continue

            elif line.startswith('----sentence'):
                sp = line[20:].lower().split()
                for w in sp:
                    if word_valid(w):
                        if w not in self.Ids:
                            # word serial
                            word_id = len(self.Ids)
                            self.Ids[w] = word_id
                            self.WordIds[w] = word_id
                            self.NodeTypes[word_id] = self.TypeIds['word']
                        else:
                            word_id = self.Ids[w]

                        # word count
                        self.WordCount[word_id] = self.WordCount.get(word_id,0) + 1
                        # word DFs
                        if word_id not in self.WordDFs:
                            self.WordDFs[word_id] = set()
                        if doc_id not in self.WordDFs[word_id]:
                            self.WordDFs[word_id].add(doc_id)
                        # DWD links
                        self.Links.append((doc_id, self.RelationIds['document_word'], word_id))

            # linkage from KB
            elif line.startswith('\t\t'):
                head = sp[0]
                tail = sp[2]
                relation = sp[1]
                # register Ids
                if head not in self.Ids:
                    self.Ids[head] = len(self.Ids)
                head_id = self.Ids[head]
                if tail not in self.Ids:
                    self.Ids[tail] = len(self.Ids)
                tail_id = self.Ids[tail]
                if relation not in self.RelationIds:
                    self.RelationIds[relation] = len(self.RelationIds)
                    self.RelationIds['_' + relation] = len(self.RelationIds)
                relation_id = self.RelationIds[relation]

                # add relation links
                self.Links.append((head_id, relation_id, tail_id))

            # named entities
            elif line.startswith('\t'):
                # register Ids
                entity = sp[0]
                entity_type = sp[1]
                if entity not in self.Ids:
                    self.Ids[entity] = len(self.Ids)
                entity_id = self.Ids[entity]
                if entity_type not in self.TypeIds:
                    self.TypeIds[entity_type] = len(self.TypeIds)
                entity_typeId = self.TypeIds[entity_type]
                self.NodeTypes[entity_id] = entity_typeId

                # entity DFs
                if entity_id not in self.EntityDFs:
                    self.EntityDFs[entity_id] = set()
                if doc_id not in self.EntityDFs[entity_id]:
                    self.EntityDFs[entity_id].add(doc_id)

                # Document-Entity links
                relation = 'document_' + entity_type
                if relation not in self.RelationIds:
                    self.RelationIds[relation] = len(self.RelationIds)
                    self.RelationIds['_' + relation] = len(self.RelationIds)
                relation_id = self.RelationIds[relation]
                self.Links.append((doc_id, relation_id, entity_id))

        # calculate word DF
        df = {}
        for word_id in self.WordDFs:
            df[word_id] = len(self.WordDFs[word_id])
        self.WordDFs = df

        # calculate entity DF
        df = {}
        for entity_id in self.EntityDFs:
            df[entity_id] = len(self.EntityDFs[entity_id])
        self.EntityDFs = df

        print 'word max count %d' % max(self.WordCount, key =self.WordCount.get)
        print 'word max DF %d' % max(df, key=df.get)

    def getlabel(self):
        return self.DocLabels

    def getHIN(self):
        hin = HIN()
        hin.Ids = self.Ids
        hin.DocIds = self.DocIds
        hin.DocLabels = self.DocLabels
        hin.Links = self.Links
        hin.RelationIds = self.RelationIds
        hin.NodeTypes = self.NodeTypes
        hin.TypeIds = self.TypeIds
        hin.WordIds = self.WordIds
        hin.WordDFs = self.WordDFs
        hin.WordCount = self.WordCount
        hin.EntityDFs = self.EntityDFs
        return hin

class YAGOReader:
    def __init__(self,path='/data/YAGO/yagoFacts.tsv'):
        self.path = path

        # define tsv dialect
        self.tsvDialect = csv.Dialect()
        self.tsvDialect.delimiter = '\t'
        self.tsvDialect.delimiter = '"'

        # Attributes Declaration
        self.Ids = {}
        self.Links = []
        self.RelationIds = {}
        self.NodeTypes = {}
        self.TypeIds = {}

    def readfile(self):
        tsvfile = open(self.path)
        reader = csv.reader(tsvfile,dialect=self.tsvDialect)

        for line in reader:
            head = line[1]
            relation = line[2]
            tail = line[3]
            if head not in self.Ids:
                head_id = len(self.Ids)
                self.Ids[head] = head_id
            else:
                head_id = self.Ids[head]
            if tail not in self.Ids:
                tail_id = len(self.Ids)
                self.Ids[tail] = tail_id
            else:
                tail_id = self.Ids[tail]
            if relation not in self.RelationIds:
                relation_id = len(self.RelationIds)
                self.RelationIds[relation] = relation_id
            else:
                relation_id = self.RelationIds[relation]

            self.Links.append((head_id,relation_id,tail_id))

    def getHIN(self):
        hin = HIN()
        hin.Ids = self.Ids
        hin.RelationIds = self.RelationIds
        hin.Links = self.Links

        return hin

class FB15KReader:
    def __init__(self,path='/home/data/corpora/FB15K/'):
        self.path = path

        # define tsv dialect
        self.tsvDialect = csv.Dialect()
        self.tsvDialect.delimiter = '\t'
#        self.tsvDialect.delimiter = '"'

        # Attributes Declaration
        self.Ids = {}
        self.Links = []
        self.TestLinks = []
        self.ValidLinks = []
        self.RelationIds = {}
        self.NodeTypes = {}
        self.TypeIds = {}

    def readfile(self):
        train = open(self.path + 'train.txt')
        valid = open(self.path + 'valid.txt')
        test = open(self.path + 'test.txt')
        trainReader = csv.reader(train,dialect=self.tsvDialect)
        validReader = csv.reader(valid,dialect=self.tsvDialect)
        testReader = csv.reader(test,dialect=self.tsvDialect)

        for line in trainReader:
            head = line[0]
            relation = line[1]
            tail = line[2]
            if head not in self.Ids:
                head_id = len(self.Ids)
                self.Ids[head] = head_id
            else:
                head_id = self.Ids[head]
            if tail not in self.Ids:
                tail_id = len(self.Ids)
                self.Ids[tail] = tail_id
            else:
                tail_id = self.Ids[tail]
            if relation not in self.RelationIds:
                relation_id = len(self.RelationIds)
                self.RelationIds[relation] = relation_id
            else:
                relation_id = self.RelationIds[relation]
            self.Links.append((head_id,relation_id,tail_id))

        for line in validReader:
            head = line[0]
            relation = line[1]
            tail = line[2]
            if head not in self.Ids:
                head_id = len(self.Ids)
                self.Ids[head] = head_id
            else:
                head_id = self.Ids[head]
            if tail not in self.Ids:
                tail_id = len(self.Ids)
                self.Ids[tail] = tail_id
            else:
                tail_id = self.Ids[tail]
            if relation not in self.RelationIds:
                relation_id = len(self.RelationIds)
                self.RelationIds[relation] = relation_id
            else:
                relation_id = self.RelationIds[relation]
            self.ValidLinks.append((head_id,relation_id,tail_id))

        for line in testReader:
            head = line[0]
            relation = line[1]
            tail = line[2]
            if head not in self.Ids:
                head_id = len(self.Ids)
                self.Ids[head] = head_id
            else:
                head_id = self.Ids[head]
            if tail not in self.Ids:
                tail_id = len(self.Ids)
                self.Ids[tail] = tail_id
            else:
                tail_id = self.Ids[tail]
            if relation not in self.RelationIds:
                relation_id = len(self.RelationIds)
                self.RelationIds[relation] = relation_id
            else:
                relation_id = self.RelationIds[relation]
            self.TestLinks.append((head_id,relation_id,tail_id))


    def getHIN(self):
        hin = HIN()
        hin.Ids = self.Ids
        hin.RelationIds = self.RelationIds
        hin.Links = self.Links
        hin.ValidLinks= self.ValidLinks
        hin.TestLinks = self.TestLinks

        return hin

# TODO : for NELL data set, understand their type system.
# Done : NELL have no type system, the semantic is stored in links
class NELLReader:
    def __init__(self,path='/data/NELL/NELL.csv'):
        self.path = path

        # Attributes Declaration
        self.Ids = {}
        self.Links = []
        self.RelationIds = {}
        self.NodeTypes = {}
        self.TypeIds = {}

    def readfile(self):
        csvfile = open(self.path)
        reader = csv.DictReader(csvfile)
        for line in reader:
            head = line['Entity']
            relation = line['Relation']
            tail = line['Value']

            if head not in self.Ids:
                head_id = len(self.Ids)
                self.Ids[head] = head_id
            else:
                head_id = self.Ids[head]
            if tail not in self.Ids:
                tail_id = len(self.Ids)
                self.Ids[tail] = tail_id
            else:
                tail_id = self.Ids[tail]
            if relation not in self.RelationIds:
                relation_id = len(self.RelationIds)
                self.RelationIds[relation] = relation_id
            else:
                relation_id = self.RelationIds[relation]

            self.Links.append((head_id, relation_id, tail_id))

    def getHIN(self):
        hin = HIN()
        hin.Ids = self.Ids
        hin.RelationIds = self.RelationIds
        hin.Links = self.Links


# Read DBLP data from Yizhou Sun
class DBLPReader:
    def __init__(self,path='/home/data/corpora/DBLP/'):
        self.path = path

        # Attributes Declaration
        self.Ids = {}
        self.Links = []
        self.RelationIds = {}
        self.NodeTypes = {}
        self.TypeIds = {}

        # Special : Old - New Id projection from file
        self.AuthorOldId = {}
        self.ConfOldId = {}
        self.TermOldId = {}

        # Labels for semi-supervised classification
        self.AuthorLabels = {}
        self.ConfLabels = {}

        # Fixed types
        self.TypeIds['word'] = 0

        # Fixed Relations
        if 'author' not in self.TypeIds:
            self.TypeIds['author'] = len(self.TypeIds)
        if 'conference' not in self.TypeIds:
            self.TypeIds['conference'] = len(self.TypeIds)
        if 'term' not in self.TypeIds:
            self.TypeIds['term'] = len(self.TypeIds)

        if 'author_author' not in self.RelationIds:
            self.RelationIds['author_author'] = len(self.RelationIds)
        if 'author_term' not in self.RelationIds:
            self.RelationIds['author_term'] = len(self.RelationIds)
        if 'conference_author' not in self.RelationIds:
            self.RelationIds['conference_author'] = len(self.RelationIds)
        if 'conference_term' not in self.RelationIds:
            self.RelationIds['conference_term'] = len(self.RelationIds)

    def readfile(self):
        authorDictFile = open(self.path + 'author_dict.txt')
        confDictFile = open(self.path + 'conf_dict.txt')
        termDictFile = open(self.path + 'term_dict.txt')

        # read and convert Ids
        for line in authorDictFile.readlines():
            sp = line.split('\t')
            author = sp[1]
            author_old_id = int(sp[0])
            author_id = len(self.Ids)
            self.Ids[author] = author_id
            self.AuthorOldId[author_old_id] = author_id
            self.NodeTypes[author_id] = self.TypeIds['author']

        for line in confDictFile.readlines():
            sp = line.split('\t')
            conf = sp[1]
            conf_old_id = int(sp[0])
            conf_id = len(self.Ids)
            self.Ids[conf] = conf_id
            self.ConfOldId[conf_old_id] = conf_id
            self.NodeTypes[conf_id] = self.TypeIds['conference']

        for line in termDictFile.readlines():
            sp = line.split('\t')
            term = sp[1]
            term_old_id = int(sp[0])
            term_id = len(self.Ids)
            self.Ids[term] = term_id
            self.TermOldId[term_old_id] = term_id
            self.NodeTypes[term_id] = self.TypeIds['term']

        # read
        AAFile = open(self.path + 'AA.txt')
        ATFile = open(self.path + 'AT.txt')
        CAFile = open(self.path + 'CA.txt')
        CTFile = open(self.path + 'CT.txt')

        for line in AAFile.readlines():
            sp = line.split()
            head = int(sp[0])
            tail = int(sp[1])
            weight = int(sp[2])
            head_id = self.AuthorOldId[head]
            tail_id = self.AuthorOldId[tail]
            relation_id = self.RelationIds['author_author']
            self.Links.append((head_id, relation_id, tail_id, weight))

        for line in ATFile.readlines():
            sp = line.split()
            head = int(sp[0])
            tail = int(sp[1])
            weight = int(sp[2])
            head_id = self.AuthorOldId[head]
            tail_id = self.TermOldId[tail]
            relation_id = self.RelationIds['author_term']
            self.Links.append((head_id, relation_id, tail_id, weight))

        for line in CAFile.readlines():
            sp = line.split()
            head = int(sp[0])
            tail = int(sp[1])
            weight = int(sp[2])
            head_id = self.ConfOldId[head]
            tail_id = self.AuthorOldId[tail]
            relation_id = self.RelationIds['conference_author']
            self.Links.append((head_id, relation_id, tail_id, weight))

        for line in CTFile.readlines():
            sp = line.split()
            head = int(sp[0])
            tail = int(sp[1])
            weight = int(sp[2])
            head_id = self.ConfOldId[head]
            tail_id = self.TermOldId[tail]

            relation_id = self.RelationIds['conference_term']
            self.Links.append((head_id, relation_id, tail_id, weight))

        authorLabelFile = open(self.path + 'author_label.txt')
        confLabelFile = open(self.path + 'conf_label.txt')

        for line in authorLabelFile.readlines():
            sp = line.split()
            author = int(sp[0])
            label = int(sp[1])
            self.AuthorLabels[self.AuthorOldId[author]] = label

        for line in confLabelFile.readlines():
            sp = line.split()
            conf = int(sp[0])
            label = int(sp[2])
            self.ConfLabels[self.ConfOldId[conf]] = label

    def getHIN(self):
        hin = HIN()
        hin.Ids = self.Ids
        hin.RelationIds = self.RelationIds
        hin.Links = self.Links
        hin.NodeTypes = self.NodeTypes
        hin.TypeIds = self.TypeIds
        hin.AuthorLabels = self.AuthorLabels
        hin.ConfLabels = self.ConfLabels
        return hin

# Read four_area data from Yizhou Sun
class FourAreaReader:
    def __init__(self,path='/home/data/corpora/four_area/'):
        self.path = path

        # Attributes Declaration
        self.Ids = {}
        self.Links = []
        self.RelationIds = {}
        self.NodeTypes = {}
        self.TypeIds = {}

        # Special : Old - New Id projection from file
        self.AuthorOldId = {}
        self.ConfOldId = {}
        self.TermOldId = {}

        # Labels for semi-supervised classification
        self.AuthorLabel = {}
        self.ConfLabel = {}

        # Fixed types
        self.TypeIds['word'] = 0

        # Fixed Relations
        if 'author' not in self.TypeIds:
            self.TypeIds['author'] = len(self.TypeIds)
        if 'conference' not in self.TypeIds:
            self.TypeIds['conference'] = len(self.TypeIds)
        if 'term' not in self.TypeIds:
            self.TypeIds['term'] = len(self.TypeIds)

        if 'author_author' not in self.RelationIds:
            self.RelationIds['author_author'] = len(self.RelationIds)
        if 'author_term' not in self.RelationIds:
            self.RelationIds['author_term'] = len(self.RelationIds)
        if 'conference_author' not in self.RelationIds:
            self.RelationIds['conference_author'] = len(self.RelationIds)
        if 'conference_term' not in self.RelationIds:
            self.RelationIds['conference_term'] = len(self.RelationIds)

    def readfile(self):
        pass
