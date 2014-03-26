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
    
    def mapChunk(self,chunk,vocabulary,typeVocabulary):
        res=[]
        for g,d,dType,count in chunk:
            g=vocabulary.get(g)
            d=vocabulary.get(d)
            tFWD,tREW=-1,-1
            if typeVocabulary!=None:
                tVFWD,tVREW=typeVocabulary.get(dType+u"-gov"),typeVocabulary.get(dType+u"-dep")
                if tVFWD!=None:
                    tFWD=tVFWD.index
                if tVREW!=None:
                    tREW=tVREW.index
            if g!=None and d!=None:
                res.append("\t".join((str(g.index),str(d.index),str(tFWD),str(count))))
                res.append("\t".join((str(d.index),str(g.index),str(tREW),str(count))))
        return "\n".join(res)

    def run(self):
        v=Vocabulary.from_pickle(self.vocabFile)
        if "typeVocabFile" in self.__dict__:
            vType=Vocabulary.from_pickle(self.typeVocabFile) #vocabulary for types
        else:
            vType=None
        #How do I open the corpus?
        if "corpusLoc" in self.__dict__ and "partsList" in self.__dict__:
            corpus=self.corpusClass.from_dir_and_part(dirName=self.corpusLoc,part=self.part,fileSegments=self.partsList)
        else:
            corpus=self.corpusClass.from_filelist(fileNames=self.fileNames)
        #And which pairs should I generate? (relevant for ngram data)
        if "whichPairs" in self.__dict__:
            it=corpus.iterPairs(whichPairs=self.whichPairs)
        else:
            it=corpus.iterPairs()
        for chunk in utils.grouper(it,self.chunkSize):
            chunkLen=len(chunk)
            chunkStr=self.mapChunk(chunk,v,vType)
            if chunkStr:
                self.lock.acquire()
                print "### ChunkRows %d Progress %f ###"%(chunkLen,corpus.progress())
                print chunkStr
                sys.stdout.flush()
                self.lock.release()



def generateSYN(corpusLoc,part,chunkSize,vocabFile,typeVocabFile=None):
    lock=mp.Lock()
    processes=[]
    for fileChunk in utils.grouper(range(100),20):
        p=NGramGenerator(corpusClass=GoogleSynNGramCorpus,corpusLoc=corpusLoc,part=part,partsList=fileChunk,lock=lock,chunkSize=chunkSize,vocabFile=vocabFile,typeVocabFile=typeVocabFile)
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
    generateSYN("/usr/share/ParseBank/google-syntax-ngrams/arcs/randomly_shuffled","arcs",50000,"../models/vocab/ENG-google-syntax-words-lookupIndex.pkl","../models/vocab/ENG-google-syntax-deptypes-lookupIndex.pkl")
    #generateFLAT("/usr/share/ParseBank/google-ngrams/5grams/repacked-uniq",50000,"../models/eng-flatng-lookupIndex.pkl","L*")
    

