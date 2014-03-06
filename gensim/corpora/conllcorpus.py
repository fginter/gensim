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

#from gensim import interfaces

logger = logging.getLogger('gensim.corpora.conllcorpus')


class CoNLLCorpus(object):
    
    def __init__(self,fName):
        if not isinstance(fName,basestring):
            raise ValueError("You need to initialize CoNLLCorpus with a fileName. It can be gzipped (.gz suffix)")
        self.fName=fName

    def iterSentences(self,lowercase_all=True,max_count=-1):
        """Generates lists of tokens"""
        if self.fName.endswith(".gz"): #...gzipped
            fIN=codecs.getreader("utf-8")(gzip.open(self.fName,"r"))
        else:
            fIN=codecs.open(self.fName,"r","utf-8")
        sentCounter=0
        currSentence=[]
        for line in fIN:
            if line.startswith(u"1\t"): #new sentence
                if currSentence:
                    yield currSentence
                    sentCounter+=1
                    if max_count>=0 and sentCounter>=max_count:
                        break #Done
                currSentence=[]
            if line and line[0].isdigit(): #Looks like a normal line, anything else is some junk or empty line -> ignore
                wordID,token,_=line.split(u"\t",2)
                currSentence.append(token)
        else:
            if currSentence:
                yield currSentence

if __name__=="__main__":
    #Quick test only
    import sys

    C=CoNLLCorpus("/usr/share/ParseBank/parsebank_v3.conll09.gz")
    for s in C.iterSentences(max_count=30):
        print (u" ".join(s).encode("utf-8"))



            
