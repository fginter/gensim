import sys
import multiprocessing as mp
from gensim.corpora.googlesynngramcorpus import GoogleSynNGramCorpus
from gensim.models.word2vec import Vocabulary, Vocab
from gensim import utils

class NGramGenerator(mp.Process):
    
    def __init__(self,corpusLoc,part,partsList,outLock,chunkSize,vocabFile):
        mp.Process.__init__(self)
        self.corpusLoc=corpusLoc
        self.part=part
        self.partsList=partsList
        self.chunkSize=chunkSize
        self.lock=outLock
        self.vocabFile=vocabFile
    
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
        corpus=GoogleSynNGramCorpus(self.corpusLoc,self.part,self.partsList)
        for chunk in utils.grouper(corpus.iterGD(),self.chunkSize):
            chunkLen=len(chunk)
            chunkStr=self.mapChunk(chunk,v)#"\n".join("\t".join(unicode(x).encode("utf-8") for x in gdtcTuple) for gdtcTuple in chunk)
            self.lock.acquire()
            print "### ChunkRows %d Progress %f ###"%(chunkLen,corpus.progress())
            print chunkStr
            sys.stdout.flush()
            self.lock.release()

def generate(corpusLoc,part,chunkSize,vocabFile):
    lock=mp.Lock()
    processes=[]
    for fileChunk in utils.grouper(range(100),10):
        p=NGramGenerator(corpusLoc,part,fileChunk,lock,chunkSize,vocabFile)
        processes.append(p)
        p.start()
    

if __name__=="__main__":
    import os
    os.nice(19)
    generate("/usr/share/ParseBank/google-syntax-ngrams/arcs/randomly_shuffled","arcs",50000,"../models/eng-full-lookupIndex.pkl")

    

