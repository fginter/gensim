#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Filip Ginter <ginter@cs.utu.fi>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Corpus in a CoNLL* format. 

"""


from __future__ import with_statement

import logging
import gzip
import itertools
import codecs
import subprocess #for external gzip. Reading gzipped corpora is a bottleneck, so let's multithread it
import os.path

#from gensim import interfaces
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('gensim.corpora.conllcorpus')

class CoNLLCorpus(object):
    
    @classmethod
    def from_filelist(cls,fileNames):
        return cls(fileNames=fileNames)

    def __init__(self,fileNames):
        self.fileNames=fileNames
        self.gzBytesRead=0
        self.totalGzBytes=sum(os.path.getsize(fName) for fName in self.fileNames)

    def iterSentences(self,column=1,lowercase_all=True,attachPos=False):
        """Generates lists of tokens from a CoNLL file. UTF8 encoding pretty much assumed.
        `column` = which column holds the tokens? 1 is the default, but you might also want 2 or 3 for lemma
        `lowercase_all` = should we lowercase all tokens for you?
        """
        for sent in self._parseSentences(tokenColumn=column,lowercase_all=lowercase_all):
            if attachPos>0:
                yield [t[0]+u"/"+t[1] for t in sent]
            else:
                yield [t[0] for t in sent]

    def progress(self):
        """A [0,1] value reflecting the progress through the corpus in terms of (compressed) bytes read."""
        return float(self.gzBytesRead)/self.totalGzBytes

    #A B C D E F G H
    #0 1 2 3 4 5 6 7
    def iterFlatNGrams(self,N,tokenColumn=1,attachPos=False):
        """Yields (flat) ngrams from the text"""
        sentCounter=0
        for sent in self.iterSentences(column=tokenColumn,lowercase_all=True,attachPos=attachPos):
            if len(sent)<N:
                continue
            for t1 in xrange(len(sent)-N+1): #first word
                yield u" ".join(itertools.islice(sent,t1,t1+N))
            sentCounter+=1
            if sentCounter%10000==0:
                logger.info("Processed %d sentences (%.3f%%)"%(sentCounter,self.progress()*100.0))

    def _parseSentences(self,tokenColumn=1,headColumn=9,deprelColumn=11,posColumn=4,lowercase_all=True):
        """Generates one sentence at a time, each sentence being a list of (token,int(head)-1,dType) tuples from the data.
        `tokenColumn` = from which column we should take the token text?
        `headColumn` = which column codes the head position?
        `deprelColumn` = which column codes the deprel?
        `lowercase_all` = Lowercase all tokens?
        """

        gzBytesReadCompleteFiles=0 #bytes read from *completed* files
        for fName in self.fileNames:
            fIN=gzip.open(fName,"r")
            currSentence=[]
            for line in fIN:
                self.gzBytesRead=fIN.myfileobj.tell()+gzBytesReadCompleteFiles #set the position in the collection
                if line.startswith("1\t"): #new sentence
                    if currSentence:
                        yield currSentence
                    currSentence=[]
                if line and line[0].isdigit(): #Looks like a normal line, anything else is some junk or empty line -> ignore
                    cols=line.split("\t",deprelColumn+1) #split as many times as needed to get to the deprel (which is always the right-most)
                    if lowercase_all:
                        currSentence.append((unicode(cols[tokenColumn],"utf-8").lower(),cols[posColumn],int(cols[headColumn])-1,cols[deprelColumn]))
                    else:
                        currSentence.append((unicode(cols[tokenColumn],"utf-8"),cols[posColumn],int(cols[headColumn])-1,cols[deprelColumn]))
            else:
                if currSentence:
                    yield currSentence
            fIN.close()
            gzBytesReadCompleteFiles+=os.path.getsize(fName)

    def iterPairs(self,tokenColumn=1,headColumn=9,deprelColumn=11,lowercase_all=True):
        """Generates (gov,dep,dType,weight) tuples from the data.
        `tokenColumn` = from which column we should take the token text?
        `headColumn` = which column codes the head position?
        `deprelColumn` = which column codes the deprel?
        `lowercase_all` = Lowercase all tokens?
        """
        for sent in self._parseSentences(tokenColumn, headColumn, deprelColumn,lowercase_all):
            for dependent, headPos, depType in sent: #headPos is 0-based index into sent to the head token. -1 for root
                if headPos==-1:
                    continue #TODO: How should I deal with the root?
                yield (sent[headPos][0],dependent,depType,1.0) #one dependency

if __name__=="__main__":
    #Quick test only
    import sys

    C=CoNLLCorpus.from_filelist(["/usr/share/ParseBank/parsebank_v3.conll09.gz"])
    for x in C.iterFlatNGrams(5,attachPos=True):
        print x.encode("utf-8")


