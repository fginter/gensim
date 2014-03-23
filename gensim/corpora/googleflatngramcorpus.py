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

#from gensim import interfaces

logger = logging.getLogger('gensim.corpora.googleflatngramcorpus')

class GoogleFlatNGramCorpus(object):
    """This class is for reading the ver 2 ngram corpus here:
    http://storage.googleapis.com/books/ngrams/books/datasetsv2.html
    """
    
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

    def lines(self,fileCount=-1):
        """
        Yields lines from the first `fileCount` files in the corpus as unicode strings.
        `fileCount` = How many files to visit? Set to -1 for all.
        """
        gzBytesReadCompleteFiles=0 #bytes read from *completed* files
        if fileCount==-1:
            fileCount=len(self.fileNames)
        for fName in self.fileNames[:fileCount]:
            with gzip.open(fName,"r") as fIN:
                for ngramLine in fIN:
                    ngramLine=unicode(ngramLine.strip(),"utf-8") #strip and skip over (possible) empty lines
                    if not ngramLine:
                        continue
                    self.gzBytesRead=fIN.myfileobj.tell()+gzBytesReadCompleteFiles #set the position in the collection
                    yield ngramLine
            gzBytesReadCompleteFiles+=os.path.getsize(fName)

    def yieldPairs(self,nGram,count,whichPairs):
        """" Yields (left,right,whichPairs,count) from the nGram. Here
        whichPairs sits in the same spot as dependency type for the
        syntactic ngram corpus, as it might be useful for something
        downstream.

        `nGram` is a unicode
        `count` is an integer
        `whichPairs` = L* for leftmost word against all, *R for rightmost word against all, LR for (leftmostword,rightmostword), and L for leftmost word only (unigram)
        """
        tokens=nGram.split()
        if u"_" in tokens[0]:
            return #Skip over the POS-labeled 
        if whichPairs=="L*":
            l=tokens[0]
            for r in itertools.islice(tokens,1,len(tokens)):
                yield (l,r,whichPairs,count)
        elif whichPairs=="*R":
            r=tokens[-1]
            for l in itertools.islice(tokens,-1):
                yield (l,r,whichPairs,count)
        elif whichPairs=="LR":
            yield tokens[0],tokens[-1], whichPairs, count
        
    def iterPairs(self,whichPairs,fileCount=-1):
        """
        Return a generator over (word1,word2,dist,count) tuples. Dist
        1 are neighboring words w1,w2, dist -1 are neighboring words
        w2,w1.
        
        `whichPairs` = L* for leftmost word against all, *R for rightmost word against all, LR for (leftmostword,rightmostword)
        `fileCount` = How many files to visit? Set to -1 for all (default)
        """
        currNGram=None
        currCount=0
        for ngramLine in self.lines(fileCount):
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
                for x in self.yieldPairs(currNGram,currCount,whichPairs):
                    yield x
                currNGram=ngram
                currCount=count
        else:
            for x in self.yieldPairs(currNGram,currCount,whichPairs):
                yield x
            

    def iterTokens(self,fileCount=-1):
        """
        Return a generator over (token,count) tuples. This only works if part=="nodes"
        `fileCount` = How many files to visit? Set to -1 for all (default)
        """
        for l,_,_,count in self.iterPairs("LR"): #this just gives me (X,X,"LR",count) when used on unigrams 
            yield l,count

if __name__=="__main__":
    #Quick test only
    import sys
    import glob

    C=GoogleFlatNGramCorpus(fileNames=glob.glob("/usr/share/ParseBank/google-ngrams/5grams/repacked-uniq/*.gz"))
    for x in C.iterTokens():
        print x
    
#    for t in C.depTypes(3):
#        print t
#    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/arcs","arcs")
#    for g,d,t,c in C.iterGD():
#        print g.encode("utf-8"),d.encode("utf-8"),t,c
#    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/nodes","nodes")
#    for t,c in C.iterTokens():
#        print t.encode("utf-8"),c

