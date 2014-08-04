#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Filip Ginter <ginter@cs.utu.fi>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Reader for the Google N-gram corpus v2

This is currently used in n-gram based word2vec training. Maybe it will be useful elsewhere, too?
"""


from __future__ import with_statement

import logging
import itertools
import os.path
import glob
import gzip
import re
import random

#from gensim import interfaces

logger = logging.getLogger('gensim.corpora.googleflatngramcorpus')

class GoogleFlatNGramCorpus(object):
    """This class is for reading the ver 2 ngram corpus here:
    http://storage.googleapis.com/books/ngrams/books/datasetsv2.html
    """

    @classmethod
    def from_filelist(cls,fileNames):
        return cls(fileNames=fileNames)
    
    def __init__(self,fileNames=None,part=None,corpusDir=None):
        """
        Initialize the corpus from a list of file names
        
        `fileNames` = iterable of file names to work with. You can set
        this to None, in which case please provide part and corpusDir,
        whereby all files looking like
        "googlebooks-*-all-"+part+"-*.gz" will be opened.
        
        `dirName` = name of the directory (needed if fileNames==None)
        
        `part` = string which specifies which part of the corpus to
        take. Values are "extended-quadarcs", "biarcs", etc...
        
        `fileSegments` = iterable of strings like "az", used to
        restrict the reader only to certain files. Needed if
        fileNames==None.
        """
        if fileNames!=None:
            self.fileNames=list(fileNames) #make a copy
        else:
            if part not in ("1gram,2gram,3gram,4gram,5gram".split(",")):
                raise ValueError("Unknown part parameter. Use 1gram,2gram,...,5gram for the part parameter")
            self.part=part
            self.fileNames=sorted(glob.glob(os.path.join(dirName,"googlebooks-*-all-"+part+"-*.gz")))
        if not self.fileNames:
            raise ValueError("Corpus not found in %s, or no files given."%dirName)
        self.gzBytesRead=0
        self.totalGzBytes=sum(os.path.getsize(fName) for fName in self.fileNames)

    def progress(self):
        """A [0,1] value reflecting the progress through the corpus in terms of (compressed) bytes read."""
        return float(self.gzBytesRead)/self.totalGzBytes

    def lines(self):
        """
        Yields lines from the first `fileCount` files in the corpus as unicode strings.
        `fileCount` = How many files to visit? Set to -1 for all.
        """
        gzBytesReadCompleteFiles=0 #bytes read from *completed* files
        for fName in self.fileNames:
            with gzip.open(fName,"r") as fIN:
                for ngramLine in fIN:
                    ngramLine=unicode(ngramLine.strip(),"utf-8") #strip and skip over (possible) empty lines
                    if not ngramLine:
                        continue
                    self.gzBytesRead=fIN.myfileobj.tell()+gzBytesReadCompleteFiles #set the position in the collection
                    yield ngramLine
            gzBytesReadCompleteFiles+=os.path.getsize(fName)

    def yieldPairs(self,nGram,count,whichPairs,windowSize,posIntoPosition):
        """" Yields (left,right,whichPairs,count) from the nGram. Here
        whichPairs sits in the same spot as dependency type for the
        syntactic ngram corpus, as it might be useful for something
        downstream.

        `nGram` is a unicode
        `count` is an integer
        `whichPairs` = L* for leftmost word against all, *R for rightmost word against all, LR for (leftmostword,rightmostword), L**R for L* union *R
        `windowSize` = random context reduction as in Mikolov et al. Set to 0 to get all pairs, and set to integer (likely 5) to  pretend we are working
         with a `windowSize` large window (e.g. 4 for five-gram data). You can set this higher than the length of the ngrams -1 to simulate a larger window 
         (i.e. increase the chance that all tokens make it in)
        """

        tokens=nGram.split()
        POSs=[]
        if posIntoPosition:
            if posIntoPosition=="fi":
                delim="/"
            else:
                delim="_"
            tmp=[]
            for t in tokens:
                try:
                    t,POS=t.rsplit(delim,1)
                except:
                    POS="x"
                tmp.append(t)
                POSs.append(POS)
            tokens=tmp
        if u"_" in tokens[0]:
            return #Skip over the POS-labeled 

        if windowSize!=0:
            red=random.randint(0,windowSize-1) #I can reduce not at all (red==windowsSize-1), or up to leaving only one word (red==0)
            leftReduction=max(0,len(tokens)-2-red) #This is where the left context will start, max reduction (when red==0) corresponds to a context of one to the left i.e. len(tokens)-2
            red=random.randint(0,windowSize-1)
            rightReduction=min(len(tokens),red+2) #this is where the right context will end +1 (i.e. slice notation) Max reduction is one word to the right, (when red==0)
        else:
            leftReduction=0
            rightReduction=len(tokens)
        
        if whichPairs=="L*":
            l=tokens[0]
            for r in itertools.islice(tokens,1,rightReduction):
                yield (l,r,"R",count)
        elif whichPairs=="*R":
            r=tokens[-1]
            for l in itertools.islice(tokens,leftReduction,len(tokens)-1):
                yield (r,l,"L",count)
        elif whichPairs=="LR":
            yield (tokens[0],tokens[-1], "R", count)
        elif whichPairs=="L**R":
            l=tokens[0]
            for i,r in enumerate(itertools.islice(tokens,1,rightReduction)):
                if posIntoPosition:
                    yield (l,r,"R"+str(i+1)+POSs[i],count) #R1 R2 R3...
                else:
                    yield (l,r,"R"+str(i+1),count) #R1 R2 R3...
            r=tokens[-1]
            for i,l in enumerate(itertools.islice(tokens,leftReduction,len(tokens)-1)):
                if posIntoPosition:
                    yield (r,l,"L"+str(i+1)+POSs[i],count) #L1 L2 L3
                else:
                    yield (l,r,"R"+str(i+1),count) #R1 R2 R3...                    yield (r,l,"L"+str(i+1),count) #L1 L2 L3
        
    def iterPairs(self,whichPairs,windowSize,posIntoPosition):
        """
        Return a generator over (focusword,contextword,dist,count) tuples. Dist
        1 are neighboring words w1,w2, dist -1 are neighboring words
        w2,w1.
        
        `whichPairs` = L* for leftmost word against all, *R for rightmost word against all, LR for (leftmostword,rightmostword)
        `fileCount` = How many files to visit? Set to -1 for all (default)
        """
        currNGram=None
        currCount=0
        for ngramLine in self.lines():
            cols=ngramLine.split(u"\t")
            if len(cols)==4: #orig format
                ngram,year,count,bookcount=cols
            elif len(cols)==2: #my repacked format
                ngram,count=cols
            count=int(count)
            if ngram==currNGram:
                currCount+=count
            elif currNGram==None:
                currCount=count
                currNGram=ngram
            else:
                for x in self.yieldPairs(currNGram,currCount,whichPairs,windowSize,posIntoPosition):
                    yield x
                currNGram=ngram
                currCount=count
        else:
            for x in self.yieldPairs(currNGram,currCount,whichPairs,windowSize,posIntoPosition):
                yield x
            

    def iterTokens(self):
        """
        Return a generator over (token,count) tuples. This only works if part=="nodes"
        """
        for l,_,_,count in self.iterPairs("LR",0): #this just gives me (X,X,"LR",count) when used on unigrams 
            yield l,count

if __name__=="__main__":
    #Quick test only
    import sys
    import glob

    fN=glob.glob("/usr/share/ParseBank/google-ngrams/5grams/repacked-pos-uniq/*.gz")
    random.shuffle(fN)
    C=GoogleFlatNGramCorpus(fileNames=fN)
    for x in C.iterPairs("L**R",5,posIntoPosition="eng"):
        print x[2]
    
#    for t in C.depTypes(3):
#        print t
#    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/arcs","arcs")
#    for g,d,t,c in C.iterGD():
#        print g.encode("utf-8"),d.encode("utf-8"),t,c
#    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/nodes","nodes")
#    for t,c in C.iterTokens():
#        print t.encode("utf-8"),c

