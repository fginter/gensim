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
import re

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
    
    @classmethod
    def from_filelist(cls,fileNames):
        c=cls(fileNames=fileNames)
        return c
    
    @classmethod
    def from_glob(cls,globExpr):
        fNames=sorted(glob.glob(globExpr))
        return cls.from_filelist(fNames)

    @classmethod
    def from_dir_and_part(cls,dirName,part,fileSegments=None):
        """
        Initialize the corpus from a directory which contains the .gz files
        `dirName` = name of the directory
        `part` = string which specifies which part of the corpus to take. Values are "extended-quadarcs", "biarcs", etc...
        `fileSegments` = iterable of integers or None. If given, lists which X's from the "X-of-N" files are to be taken. None -> all.
        """
        if part not in ("nodes,arcs,biarcs,triarcs,quadarcs,extended-arcs,extended-biarcs,extended-triarcs,extended-quadarcs".split(",")):
                raise ValueError("Unknown part parameter. Use arcs,biarcs,...,extended-arcs,extended-biarcs.")
        fileNames=sorted(glob.glob(os.path.join(dirName,part+".*-of-*.gz")))
        if not fileNames:
            raise ValueError("Corpus not found. No files like %s.*-of-*.gz in the directory %s."%(part,dirName))
        if fileSegments!=None:
            fileSegments=set(fileSegments)
            filteredFileNames=[]
            for fName in fileNames:
                match=re.search(part+r"\.([0-9]+)-of-[0-9]+\.gz$",fName)
                if not match:
                    raise ValueError("Cannot parse filename %s - needed to restrict corpus files."%fName)
                if int(match.group(1)) in fileSegments:
                    filteredFileNames.append(fName)
            fileNames=filteredFileNames
        return cls.from_filelist(fileNames)

    def __init__(self,fileNames):
        self.fileNames=fileNames[:]
        self.gzBytesRead=0
        self.totalGzBytes=sum(os.path.getsize(fName) for fName in self.fileNames)
    
    def progress(self):
        """A [0,1] value reflecting the progress through the corpus in terms of (compressed) bytes read."""
        return float(self.gzBytesRead)/self.totalGzBytes

    def lines(self):
        """
        Yields lines from the corpus as unicode strings.
        `fileCount` = How many files to visit? Set to -1 for all.
        """
        gzBytesReadCompleteFiles=0 #bytes read from *completed* files
        for fName in self.fileNames:
            with gzip.open(fName,"r") as fIN:
                for ngramLine in fIN:
                    ngramLine=unicode(ngramLine.rstrip(),"utf-8") #strip and skip over (possible) empty lines
                    if not ngramLine:
                        continue
                    self.gzBytesRead=fIN.myfileobj.tell()+gzBytesReadCompleteFiles #set the position in the collection
                    yield ngramLine
            gzBytesReadCompleteFiles+=os.path.getsize(fName)

    def depTypes(self):
        """
        Return an iterator over dependency types in the first `analyzeFileCount` .gz files from the corpus.
        `analyzeFileCount` = How many .gz files to go through? Few will usually suffice unless you want exact stats (defaults to 5). Set to -1 if you want all.
        """
        for ngramLine in self.lines():
            #lying<TAB>lying/VBG/dobj/0 and/CC/cc/1 dying/VBG/conj/1 thinking/VBG/dep/3<TAB>12<TAB>...
            rootToken, tree, count, rest=ngramLine.split(u"\t",3) 
            count=int(count)
            for token in tree.split():
                #lying/VBG/dobj/0
                rest,depType,headNumber=token.rsplit(u"/",2)
                yield depType, count

    def iterPairs(self):
        """
        Return an iterator over (governor,dependent,depType,count) tuples. This currently only works if part=="arcs"
        `fileCount` = How many files to visit? Set to -1 for all (default)
        """
        for ngramLine in self.lines():
            #includes<TAB>includes/VBZ/rcmod/0 telecom/NNP/dobj/1<TAB>count<TAB>...
            rootToken,dependency,count,rest=ngramLine.split(u"\t",3)
            count=int(count)
            tokens=[SynTreeNode(t) for t in dependency.split()]
            if len(tokens)>2: #This involves one of the functional words, skip (we will get the relevant deps elsewhere in the data)
                continue
            tokens.sort() #Sorts them by index of the governor, which brings the root as the first token
            assert tokens[0].token==rootToken and tokens[0].governorIDX==0 and tokens[1].governorIDX in (1,2), ngramLine #any surprises somewhere?
            yield (tokens[0].token,tokens[1].token,tokens[1].dType,count)

    def iterTokens(self):
        """
        Return a generator over (token,count) tuples. This only works if part=="nodes"
        `fileCount` = How many files to visit? Set to -1 for all (default)
        """
        #if self.part not in (u"nodes",):
        #    raise ValueError("Tokens can be generated only from the 'nodes' part of the corpus. See GoogleSynNGramCorpus.__init__()")
        currentToken=None
        currentCount=0
        for ngramLine in self.lines():
            #for every token, we'll have a series of lines like this:
            #bookers<TAB>bookers/NNP/ROOT/0<TAB>count<TAB>...      1882,1...
            cols=ngramLine.split(u"\t",3)
            if len(cols)!=4: #skipping over some weird whitespace token cases
                continue
            token,specs,count,rest=cols
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
    
    C=GoogleSynNGramCorpus.from_glob("/usr/share/ParseBank/google-syntax-ngrams/arcs/*.gz")
    for x in C.iterPairs():
        print x
    
#    for t in C.depTypes():
#        print t
#    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/arcs","arcs")
#    for g,d,t,c in C.iterGD():
#        print g.encode("utf-8"),d.encode("utf-8"),t,c
#    C=GoogleSynNGramCorpus("/usr/share/ParseBank/google-syntax-ngrams/nodes","nodes")
#    for t,c in C.iterTokens():
#        print t.encode("utf-8"),c

