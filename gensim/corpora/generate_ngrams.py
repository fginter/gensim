import sys
import multiprocessing as mp
from gensim.corpora.googlesynngramcorpus import GoogleSynNGramCorpus
from gensim import utils

class NGramGenerator(mp.Process):
    
    def __init__(self,corpusLoc,part,partsList,outLock,chunkSize):
        mp.Process.__init__(self)
        self.corpusLoc=corpusLoc
        self.part=part
        self.partsList=partsList
        self.chunkSize=chunkSize
        self.lock=outLock

    def run(self):
        corpus=GoogleSynNGramCorpus(self.corpusLoc,self.part,self.partsList)
        for chunk in utils.grouper(corpus.iterGD(),self.chunkSize):
            chunkLen=len(chunk)
            chunkStr="\n".join("\t".join(unicode(x).encode("utf-8") for x in gdtcTuple) for gdtcTuple in chunk)
            self.lock.acquire()
            print "### ChunkRows %d Progress %f ###"%(chunkLen,corpus.progress())
            print chunkStr
            sys.stdout.flush()
            self.lock.release()

def generate(corpusLoc,part,chunkSize,processes):
    lock=mp.Lock()
    processes=[]
    for fileChunk in utils.grouper(range(100),15):
        p=NGramGenerator(corpusLoc,part,fileChunk,lock,chunkSize)
        processes.append(p)
        p.start()
    

if __name__=="__main__":
    generate("/usr/share/ParseBank/google-syntax-ngrams/arcs","arcs",50000,None)
    

