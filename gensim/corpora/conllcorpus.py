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

logger = logging.getLogger('gensim.corpora.conllcorpus')

def openGZ(fName,threads=2):
    """Tries to open a .gz file using 1) pigz 2) gzip 3) the gzip
    module, in this order of priority. If pigz is installed, it will
    be given `threads` as the number of threads to use for a further
    speed-up over normal gzip.

    `threads` = number of threads for pigz, ignored otherwise
    """
    if not os.path.exists(fName):
        raise ValueError("No such file: "+fName)
    #Try pigz (multithreaded implementation of gzip, sadly absent on most machines)
    try:
        p=subprocess.Popen(("pigz","--decompress","--to-stdout","--processes",str(threads),fName),stdout=subprocess.PIPE,stdin=None,stderr=subprocess.PIPE)
        return p.stdout
    except:
        pass
    #No pigz, try gzip
    try:
        p=subprocess.Popen(("gzip","--decompress","--to-stdout",fName),stdout=subprocess.PIPE,stdin=None,stderr=subprocess.PIPE)
        return p.stdout
    except:
        pass
    #No gzip either, too bad. This should then work for sure:
    return gzip.open(fName,"r")


class CoNLLCorpus(object):
    
    def __init__(self,fName):
        if not isinstance(fName,basestring):
            raise ValueError("You need to initialize CoNLLCorpus with a fileName. It can be gzipped (.gz suffix)")
        self.fName=fName

    def iterSentences(self,column=1,lowercase_all=True,max_count=-1,threads=2):
        """Generates lists of tokens from a CoNLL file. UTF8 encoding pretty much assumed.
        `column` = which column holds the tokens? 1 is the default, but you might also want 2 or 3 for lemma
        `lowercase_all` = should we lowercase all tokens for you?
        `max_count` = how many sentences to read?
        `threads` = if we succeed in opening a gzipped file using a multi-threaded gzip, how many threads can we give it?
        """
        for sent in self._parseSentences(tokenColumn=column,lowercase_all=lowercase_all,max_count=max_count,threads=threads):
            yield [t[0] for t in sent]

    def _parseSentences(self,tokenColumn=1,headColumn=9,deprelColumn=11,lowercase_all=True,max_count=-1,threads=2):
        """Generates one sentence at a time, each sentence being a list of (token,int(head)-1,dType) tuples from the data.
        `tokenColumn` = from which column we should take the token text?
        `headColumn` = which column codes the head position?
        `deprelColumn` = which column codes the deprel?
        `lowercase_all` = Lowercase all tokens?
        `max_count` = how many dependencies (not sentences!) should be read? -1 for all
        `threads` = if we succeed in opening a gzipped file using a multi-threaded gzip, how many threads can we give it?
        """
        if self.fName.endswith(".gz"): #...gzipped
            dataIN=openGZ(self.fName)
        else:
            dataIN=open(self.fName,"rt")
        sentCounter=0
        currSentence=[]
        for line in dataIN:
            if line.startswith("1\t"): #new sentence
                if currSentence:
                    yield currSentence
                    sentCounter+=1
                    if max_count>=0 and sentCounter>=max_count:
                        break #Done
                currSentence=[]
            if line and line[0].isdigit(): #Looks like a normal line, anything else is some junk or empty line -> ignore
                cols=line.split("\t",deprelColumn+1) #split as many times as needed to get to the deprel (which is always the right-most)
                if lowercase_all:
                    currSentence.append((unicode(cols[tokenColumn],"utf-8").lower(),int(cols[headColumn])-1,cols[deprelColumn]))
                else:
                    currSentence.append((unicode(cols[tokenColumn],"utf-8"),int(cols[headColumn])-1,cols[deprelColumn]))
        else:
            if currSentence:
                yield currSentence
        dataIN.close()


    def iterDeps(self,tokenColumn=1,headColumn=9,deprelColumn=11,lowercase_all=True,max_count=-1,threads=2):
        """Generates (gov,dep,dType,weight) tuples from the data.
        `tokenColumn` = from which column we should take the token text?
        `headColumn` = which column codes the head position?
        `deprelColumn` = which column codes the deprel?
        `lowercase_all` = Lowercase all tokens?
        `max_count` = how many sentences (not dependencies!) should be read? -1 for all
        `threads` = if we succeed in opening a gzipped file using a multi-threaded gzip, how many threads can we give it?
        """
        for sent in self._parseSentences(tokenColum, headColumn, deprelColumn,lowercase_all,max_count,threads=threads):
            for dependent, headPos, depType in sent: #headPos is 0-based index into sent to the head token. -1 for root
                if headPos==-1:
                    continue #TODO: How should I deal with the root?
                yield (sent[headPos][0],dependent,depType,1.0) #one dependency

if __name__=="__main__":
    #Quick test only
    import sys

    C=CoNLLCorpus("/usr/share/ParseBank/parsebank_v3.conll09.gz")
    for s in C.iterSentences(column=4,lowercase_all=True,max_count=30):
        print (u" ".join(s).encode("utf-8"))

