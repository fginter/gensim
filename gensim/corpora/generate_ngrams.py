import sys
import multiprocessing as mp
from gensim.corpora.googlesynngramcorpus import GoogleSynNGramCorpus
from gensim.corpora.googleflatngramcorpus import GoogleFlatNGramCorpus
from gensim.models.word2vec import Vocabulary, Vocab
from gensim import utils
import glob

class NGramGenerator(mp.Process):

    def __init__(self,corpusClass,**kwargs):
        """
        `corpusClass` either GoogleSynNGramCorpus or GoogleFlatNGramCorpus
        """
        mp.Process.__init__(self)
        self.corpusClass=corpusClass
        for k,v in kwargs.iteritems():
            self.__dict__[k]=v
    
    def mapChunk(self,chunk,vocabulary):
        res=[]
        for g,d,dType,count in chunk:
            g=vocabulary.get(g)
            d=vocabulary.get(d)
            if g!=None and d!=None:
                res.append("\t".join((str(g.index),str(d.index),dType,str(count))))
        return "\n".join(res)

    def run(self):
        v=Vocabulary.from_pickle(self.vocabFile)
        if "corpusLoc" in self.__dict__ and "partsList" in self.__dict__:
            corpus=self.corpusClass(self.corpusLoc,self.part,self.partsList)
        else:
            corpus=self.corpusClass(fileNames=self.fileNames)
        if "whichPairs" in self.__dict__:
            it=corpus.iterPairs(whichPairs=self.whichPairs)
        else:
            it=corpus.iterPairs()
        for chunk in utils.grouper(it,self.chunkSize):
            chunkLen=len(chunk)
            chunkStr=self.mapChunk(chunk,v)
            if chunkStr:
                self.lock.acquire()
                print "### ChunkRows %d Progress %f ###"%(chunkLen,corpus.progress())
                print chunkStr
                sys.stdout.flush()
                self.lock.release()

def generateSYN(corpusLoc,part,chunkSize,vocabFile):
    lock=mp.Lock()
    processes=[]
    for fileChunk in utils.grouper(range(100),10):
        p=NGramGenerator(GoogleSynNGramCorpus,corpusLoc,part,fileChunk,lock,chunkSize,vocabFile)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def generateFLAT(corpusLoc,chunkSize,vocabFile,whichPairs):
    lock=mp.Lock()
    fileNames=sorted(glob.glob(os.path.join(corpusLoc,"*.gz")))
    processes=[]
    for fileName in fileNames: #One process per file in this case TODO: group?
        p=NGramGenerator(GoogleFlatNGramCorpus,lock=lock,chunkSize=chunkSize,vocabFile=vocabFile,fileNames=[fileName],whichPairs=whichPairs)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__=="__main__":
    import os
    os.nice(19)
    #generateSYN("/usr/share/ParseBank/google-syntax-ngrams/arcs/randomly_shuffled","arcs",50000,"../models/eng-full-lookupIndex.pkl")
    generateFLAT("/usr/share/ParseBank/google-ngrams/5grams/repacked-uniq",50000,"../models/eng-flatng-lookupIndex.pkl","L*")
    

