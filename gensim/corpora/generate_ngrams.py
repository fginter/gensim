import sys
import multiprocessing as mp
from gensim.corpora.googlesynngramcorpus import GoogleSynNGramCorpus
from gensim.corpora.googleflatngramcorpus import GoogleFlatNGramCorpus
from gensim.models.word2vec import Vocabulary, Vocab
from gensim import utils
import glob
import time
import os.path

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
            pr=corpus.progress()
            self.progressArray[self.processIDX]=pr
            while min(x for x in self.progressArray)+0.01<pr: #There's someone lagging behind by more than 1pp, wait for them
                print >> sys.stderr, min(self.progressArray)
                time.sleep(3) #...wait 3 sec
            with self.progressArray.get_lock(): #use the progressArray lock to synchronize stdout
                if chunkStr:
                    print "### ChunkRows %d Progress %f ###"%(chunkLen,pr)
                    print chunkStr
                    sys.stdout.flush()

def generateSYN(corpusLoc,chunkSize,vocabFile,progressArray,processList,typeVocabFile=None):
    fileNames=sorted(glob.glob(os.path.join(corpusLoc,"*.gz")))
    perGroup=len(fileNames)//5
    for fileList in utils.grouper(fileNames,perGroup):
        p=NGramGenerator(corpusClass=GoogleSynNGramCorpus,fileNames=fileList,progressArray=progressArray,processIDX=len(processList),chunkSize=chunkSize,vocabFile=vocabFile,typeVocabFile=typeVocabFile)
        processList.append(p)
        p.start()
    return processList


def generateFLAT(corpusLoc,chunkSize,vocabFile,whichPairs,progressVal=None):
    if progressVal==None:
        progressVal=mp.Value("f") #value to share the global progress, and also synchronize the output
        progressVal.value=1.0 #the processes will rewrite this to the real value pretty much right away
    fileNames=sorted(glob.glob(os.path.join(corpusLoc,"*.gz")))
    processes=[]
    for fileName in fileNames: #One process per file in this case TODO: group?
        p=NGramGenerator(GoogleFlatNGramCorpus,progressVal=progressVal,chunkSize=chunkSize,vocabFile=vocabFile,fileNames=[fileName],whichPairs=whichPairs)
        processes.append(p)
        p.start()
    return processes
if __name__=="__main__":
    import os
    os.nice(19)
    progressArray=mp.Array("f",50) #I will hardly ever run more than 50
    for i in range(len(progressArray)):
        progressArray[i]=2.0 #impossibly high value
    processList=[]
    #processes+=generateSYN("/usr/share/ParseBank/google-syntax-ngrams/arcs/randomly_shuffled","arcs",50000,"../models/vocab/ENG-google-syntax-words-lookupIndex.pkl","../models/vocab/ENG-google-syntax-deptypes-lookupIndex.pkl")
    

    lang="eng"

    if lang=="fin":
        generateSYN("/mnt/ssd/w2v_sng_training/arcs-repacked-uniq",50000,"../models/vocab/FIN-pbv3-syntax-words-lookupIndex.pkl",progressArray,processList,typeVocabFile="../models/vocab/FIN-pbv3-syntax-deptypes-lookupIndex.pkl")
    elif lang=="eng":
        generateSYN("/usr/share/ParseBank/google-syntax-ngrams/arcs/randomly_shuffled",50000,"../models/vocab/ENG-google-syntax-words-lookupIndex.pkl",progressArray,processList,typeVocabFile="../models/vocab/ENG-google-syntax-deptypes-lookupIndex.pkl")

    for p in processList:
        p.join()

    
    

