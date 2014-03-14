#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Filip Ginter <ginter@cs.utu.fi>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Reader for the Google Syntactic N-gram corpus 

http://googleresearch.blogspot.fi/2013/05/syntactic-ngrams-over-time.html
http://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html

This is currently in n-gram based word2vec training. Maybe it will be useful elsewhere, too?
"""


from __future__ import with_statement

import logging
import itertools
import os.path
import glob
import gzip

#from gensim import interfaces

logger = logging.getLogger('gensim.corpora.googlesynngramcorpus')

class SynTreeNode(object):
    
    """Represents one node (word) in the syntactic tree."""

    def __init__(self,descString):
        """`descString` one token descriptor like u'lying/VBG/dobj/0'"""
        self.token,self.pos,self.dType,self.governorIDX=descString.rsplit(u"/",3)
        self.governorIDX=int(self.governorIDX)

    def __cmp__(self,other):
        """Nodes sort on their governor index, which induces a topological order"""
        return cmp(self.governorIDX,other.governorIDX)

class GoogleSynNGramCorpus(object):
    
    def __init__(self,dirName,part):
        """
        Initialize the corpus from a directory which contains the .gz files
        `dirName` = name of the directory
        `part` = string which specifies which part of the corpus to take. Values are "extended-quadarcs", "biarcs", etc...
        """
        if part not in ("nodes,arcs,biarcs,triarcs,quadarcs,extended-arcs,extended-biarcs,extended-triarcs,extended-quadarcs".split(",")):
            raise ValueError("Unknown part parameter. Use arcs,biarcs,...,extended-arcs,extended-biarcs.")
        self.part=part
        self.fileNames=sorted(glob.glob(os.path.join(dirName,part+".*-of-*.gz")))
        if not self.fileNames:
            raise ValueError("Corpus not found. No files like %s.*-of-*.gz in the directory %s."%(part,dirName))
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

    def depTypes(self,analyzeFileCount=5):
        """
        Return an iterator over dependency types in the first `analyzeFileCount` .gz files from the corpus.
        `analyzeFileCount` = How many .gz files to go through? Few will usually suffice unless you want exact stats (defaults to 5). Set to -1 if you want all.
        """
        for ngramLine in self.lines(analyzeFileCount):
            #lying<TAB>lying/VBG/dobj/0 and/CC/cc/1 dying/VBG/conj/1 thinking/VBG/dep/3<TAB>12<TAB>...
            rootToken, tree, rest=ngramLine.split(u"\t",2) 
            for token in tree.split():
                #lying/VBG/dobj/0
                rest,depType,headNumber=token.rsplit(u"/",2)
                yield depType

    def iterGD(self,fileCount=-1):
        """
        Return a generator over (governor,dependent,depType,count) tuples. This currently only works if part=="arcs"
        `fileCount` = How many files to visit? Set to -1 for all (default)
        """
        if self.part not in (u"arcs",):
            raise ValueError("GDPairs can be generated only from the 'arcs' part of the corpus. See GoogleSynNGramCorpus.__init__()")
        for ngramLine in self.lines(fileCount):
            #includes<TAB>includes/VBZ/rcmod/0 telecom/NNP/dobj/1<TAB>count<TAB>...
            try:
                rootToken,dependency,count,rest=ngramLine.split(u"\t",3)
            except ValueError:
                rootToken,dependency,count=ngramLine.split(u"\t",2)
            count=int(count)
            tokens=[SynTreeNode(t) for t in dependency.split()]
            if len(tokens)>2: #This involves one of the functional words, skip (we will get the relevant deps elsewhere in the data)
                continue
            tokens.sort() #Sorts them by index of the governor, which brings the root as the first token
            assert tokens[0].token==rootToken and tokens[0].governorIDX==0 and tokens[1].governorIDX in (1,2), ngramLine #any surprises somewhere?
            yield (tokens[0].token,tokens[1].token,tokens[1].dType,count)

    def iterTokens(self,fileCount=-1):
        """
        Return a generator over (token,count) tuples. This only works if part=="nodes"
        `fileCount` = How many files to visit? Set to -1 for all (default)
        """
        if self.part not in (u"nodes",):
            raise ValueError("Tokens can be generated only from the 'nodes' part of the corpus. See GoogleSynNGramCorpus.__init__()")
        currentToken=None
        currentCount=0
        for ngramLine in self.lines(fileCount):
            #for every token, we'll have a series of lines like this:
            #bookers<TAB>bookers/NNP/ROOT/0<TAB>count<TAB>...      1882,1...
            try:
                token,specs,count,rest=ngramLine.split(u"\t",3)
            except ValueError:
                token,specs,count=ngramLine.split(u"\t",2)
            if specs.count(u" ")!=0: #several nodes, ignore. TODO: is this supposed to be skipped or processed?
                continue
            count=int(count)
            if currentToken==token: #...still continuing the current token?
                currentCount+=count
            elif currentToken==None: #First one?
                currentToken=token
                currentCount=count
            else: #New one!
                yield currentToken, currentCount
                currentToken=token
                currentCount=count
        else: #End of file
            if currentToken!=None:
                yield currentToken, currentCount #Remember to yield the last one
        

if __name__=="__main__":
    #Quick test only
    import sys
    
    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/quadarcs","quadarcs")
#    for t in C.depTypes(3):
#        print t
    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/arcs","arcs")
#    for g,d,t,c in C.iterGD():
#        print g.encode("utf-8"),d.encode("utf-8"),t,c
    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/nodes","nodes")
    for t,c in C.iterTokens():
        print t.encode("utf-8"),c

