import scipy.sparse as sparse

class LinkFilter():
    @staticmethod
    def filter(Links,criteria):
        filteredLinks = []
        for link in Links:
            if criteria(link):
                filteredLinks.append(link)
        return filteredLinks

    @staticmethod
    def filterByRelation(Links,relationId):
        filteredLinks = []
        for link in Links:
            if link(1) == relationId:
                filteredLinks.append(link)
        return filteredLinks

    @staticmethod
    def filterByNode(Links,nodeType):
        filteredLinks = []
        for link in Links:
            if link(0) in nodeType or link(2) in nodeType:
                filteredLinks.append(link)
        return filteredLinks

    @staticmethod
    def filterByBothNode(Links,headType,tailType):
        filteredLinks = []
        for link in Links:
            if link(0) in headType and link(2) in tailType:
                filteredLinks.append(link)
        return filteredLinks