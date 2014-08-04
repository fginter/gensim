import sys
import multiprocessing as mp
from gensim.corpora.googlesynngramcorpus import GoogleSynNGramCorpus
from gensim.corpora.googleflatngramcorpus import GoogleFlatNGramCorpus
from gensim.corpora.conllcorpus import CoNLLCorpus
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
                if dType.startswith("R") and dType[1].isdigit(): #Dealing with flat n-gram data
                    try:
                        tFWD=typeVocabulary.get(dType).index
                    except:
                        pass
                    try:
                        tREW=typeVocabulary.get(dType.replace("R","L")).index
                    except:
                        pass
                elif dType.startswith("L") and dType[1].isdigit():
                    try:
                        tFWD=typeVocabulary.get(dType).index
                    except:
                        pass
                    try:
                        tREW=typeVocabulary.get(dType.replace("L","R")).index
                    except:
                        pass
                else: #Dealing with syntax data
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
            it=corpus.iterPairs(whichPairs=self.whichPairs,windowSize=self.windowSize,posIntoPosition=self.posIntoPosition) #flat ngram data
        else:
            it=corpus.iterPairs(arcness=self.arcness,minCount=15)
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

def generateSYN(corpusLoc,chunkSize,vocabFile,progressArray,processList,arcness,typeVocabFile=None):
    fileNames=sorted(glob.glob(os.path.join(corpusLoc,"*.gz")))
    perGroup=len(fileNames)//3
    for fileList in utils.grouper(fileNames,perGroup):
        p=NGramGenerator(corpusClass=GoogleSynNGramCorpus,fileNames=fileList,progressArray=progressArray,processIDX=len(processList),chunkSize=chunkSize,vocabFile=vocabFile,typeVocabFile=typeVocabFile,arcness=arcness)
        processList.append(p)
        p.start()


def generateSYNFULL(fileList,chunkSize,vocabFile,progressArray,processList,typeVocabFile=None):
    p=NGramGenerator(corpusClass=CoNLLCorpus,fileNames=fileList,progressArray=progressArray,processIDX=len(processList),chunkSize=chunkSize,vocabFile=vocabFile,typeVocabFile=typeVocabFile)
    processList.append(p)
    p.start()

def generateFLAT(corpusLoc,chunkSize,vocabFile,progressArray,processList,whichPairs,typeVocabFile=None,posIntoPosition=False):
    fileNames=sorted(glob.glob(os.path.join(corpusLoc,"*.gz")))
    perGroup=len(fileNames)//10
    for fileList in utils.grouper(fileNames,perGroup):
        p=NGramGenerator(corpusClass=GoogleFlatNGramCorpus,fileNames=fileList,progressArray=progressArray,processIDX=len(processList),chunkSize=chunkSize,vocabFile=vocabFile,typeVocabFile=typeVocabFile,whichPairs=whichPairs,windowSize=5,posIntoPosition=posIntoPosition)
        processList.append(p)
        p.start()


if __name__=="__main__":
    import os
    os.nice(19)
    progressArray=mp.Array("f",20) #I will hardly ever run more than 20 of these TODO _ FIX
    for i in range(len(progressArray)):
        progressArray[i]=2.0 #impossibly high value
    processList=[]
    #processes+=generateSYN("/usr/share/ParseBank/google-syntax-ngrams/arcs/randomly_shuffled","arcs",50000,"../models/vocab/ENG-google-syntax-words-lookupIndex.pkl","../models/vocab/ENG-google-syntax-deptypes-lookupIndex.pkl")
    
    ##### CUTTING AT 15   DANGER ####################

    lang="eng-syn-123grams"
    if lang=="fin-flat-pos":
        generateFLAT("/home/ginter/fin5g/pos5g",50000,"../models/vocab/FIN-pbv3-syntax-words-lookupIndex.pkl",progressArray,processList,"L**R","../models/vocab/FIN-pbv3-pospositions-lookupIndex.pkl",posIntoPosition=True)
    elif lang=="fin-flat-ngram":
        generateFLAT("/home/ginter/fin5g",50000,"../models/vocab/FIN-pbv3-syntax-words-lookupIndex.pkl",progressArray,processList,"L**R","../models/vocab/ngram-numerical-positions-lookupIndex.pkl")
    elif lang=="fin-syn-ngram":
        generateSYN("/mnt/ssd/w2v_sng_training/arcs-repacked-uniq",50000,"../models/vocab/FIN-pbv3-syntax-words-lookupIndex.pkl",progressArray,processList,typeVocabFile="../models/vocab/FIN-pbv3-syntax-deptypes-lookupIndex.pkl")
    elif lang=="fin-syn-bingram":
        generateSYN("/mnt/ssd/w2v_sng_training/biarcs-repacked-uniq",50000,"../models/vocab/FIN-pbv3-syntax-words-lookupIndex.pkl",progressArray,processList,arcness="biarc",typeVocabFile="../models/vocab/FIN-pbv3-123arc-deptypes-lookupIndex.pkl")
    elif lang=="fin-syn-tringram":
        generateSYN("/mnt/ssd/w2v_sng_training/triarcs-repacked-uniq",50000,"../models/vocab/FIN-pbv3-syntax-words-lookupIndex.pkl",progressArray,processList,arcness="triarc",typeVocabFile="../models/vocab/FIN-pbv3-123arc-deptypes-lookupIndex.pkl")
    elif lang=="eng-syn-tringram":
        generateSYN("/usr/share/ParseBank/google-syntax-ngrams/triarcs",50000,"../models/vocab/ENG-google-syntax-words-lookupIndex.pkl",progressArray,processList,arcness="triarc",typeVocabFile="../models/vocab/FIN-pbv3-123arc-deptypes-lookupIndex.pkl")
    elif lang=="fin-syn-fulldata":
        generateSYNFULL(["/usr/share/ParseBank/parsebank_v3.conll09.gz"],50000,"../models/vocab/FIN-pbv3-syntax-words-lookupIndex.pkl",progressArray,processList,typeVocabFile="../models/vocab/FIN-pbv3-syntax-deptypes-lookupIndex.pkl")
    elif lang=="eng-syn-123grams":
        generateSYN("/usr/share/ParseBank/google-syntax-ngrams/arcs/randomly_shuffled",50000,"../models/vocab/ENG-google-syntax-words-lookupIndex.pkl",progressArray,processList,arcness="arc",typeVocabFile="../models/vocab/ENG-google-123arc-deptypes-lookupIndex.pkl")
        generateSYN("/usr/share/ParseBank/google-syntax-ngrams/biarcs/randomly_shuffled",50000,"../models/vocab/ENG-google-syntax-words-lookupIndex.pkl",progressArray,processList,arcness="biarc",typeVocabFile="../models/vocab/ENG-google-123arc-deptypes-lookupIndex.pkl")
        generateSYN("/usr/share/ParseBank/google-syntax-ngrams/triarcs/randomly_shuffled",50000,"../models/vocab/ENG-google-syntax-words-lookupIndex.pkl",progressArray,processList,arcness="triarc",typeVocabFile="../models/vocab/ENG-google-123arc-deptypes-lookupIndex.pkl")
    elif lang=="eng-flat":
        generateFLAT("/usr/share/ParseBank/google-ngrams/5grams/repacked-uniq",50000,"../models/vocab/ENG-google-flat-words-lookupIndex.pkl",progressArray,processList,"L**R","../models/vocab/ngram-numerical-positions-lookupIndex.pkl")
    elif lang=="eng-flat-pos":
        generateFLAT("/usr/share/ParseBank/google-ngrams/5grams/repacked-pos-uniq",50000,"../models/vocab/ENG-google-flat-words-lookupIndex.pkl",progressArray,processList,"L**R","../models/vocab/ENG-google-pospositions-lookupIndex.pkl",posIntoPosition="eng")

    for p in processList:
        p.join()

    
    

