import cPickle as pk

from classifier import SSLClassifier
from datareader import DataReaderWang
from graphgenerator import GraphGenerator

SIM_count = {'comp.graphics':1000,'comp.sys.mac.hardware':1000,'comp.os.ms-windows.misc':1000}
DIFF_count = {'rec.autos':1000,'comp.os.ms-windows.misc':1000,'sci.space':1000}
GSIM_count = {'GWEA':1014,'GDIS':2083,'GENV':499}
GDIF_count = {'GENT':1062,'GODD':1096,'GDEF':542}

def printNG20Types():
    with open('data/explore/NG20relation') as f:
        relations = pk.load(f)
    with open('data/explore/NG20types') as f:
        nodes = pk.load(f)
    for (k, v) in relations.items():
        print k, v
    for (k, v) in nodes.items():
        print k, v

