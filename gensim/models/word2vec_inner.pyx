#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp,pow
from libc.string cimport memset

from cpython cimport PyCObject_AsVoidPtr
from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t


ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*fast_sentence_ptr) (
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil

ctypedef void (*fast_sentence_ng_ptr) (
    const np.uint32_t *w1_point, const np.uint8_t *w1_code, const int w1_codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t w0_index, const np.uint32_t *depType_point, const np.uint8_t *depType_code, const int depType_codelen, const REAL_t alpha, REAL_t *work) nogil

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef fast_sentence_ptr fast_sentence
cdef fast_sentence_ng_ptr fast_sentence_ng


DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0


cdef void fast_sentence0(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>dsdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

cdef void fast_sentence1_ng (
    const np.uint32_t *w1_point, const np.uint8_t *w1_code, const int w1_codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t w0_index, const np.uint32_t *depType_point, const np.uint8_t *depType_code, const int depType_codelen, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = w0_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(w1_codelen):
        row2 = w1_point[b] * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - w1_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    for b in range(depType_codelen):
        row2 = depType_point[b] * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - depType_code[b] - f) * alpha 
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)


cdef void fast_sentence1(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)


cdef void fast_sentence2(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    for a in range(size):
        work[a] = <REAL_t>0.0
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>0.0
        for a in range(size):
            f += syn0[row1 + a] * syn1[row2 + a]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        for a in range(size):
            work[a] += g * syn1[row2 + a]
        for a in range(size):
            syn1[row2 + a] += g * syn0[row1 + a]
    for a in range(size):
        syn0[row1 + a] += work[a]


DEF MAX_SENTENCE_LEN = 1000
DEF MAX_NGRAMLIST_LEN = 100000

#ngrams: list of (w0,w1,dtype,weight) tuples
# here w0,w1,dtype are Vocab() instances
# we predict w1(+dtype) based on w0, i.e. w0 goes on the input, w1+dtype goes on the output
def train_synngram_list(model, ngrams, alpha, _work):
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *work
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef REAL_t counts[MAX_NGRAMLIST_LEN]
#    cdef np.uint32_t *points_w0[MAX_NGRAMLIST_LEN]
    cdef np.uint32_t *points_w1[MAX_NGRAMLIST_LEN]
    cdef np.uint32_t *points_depTypes[MAX_NGRAMLIST_LEN]
#    cdef np.uint8_t *codes_w0[MAX_NGRAMLIST_LEN]
    cdef np.uint8_t *codes_w1[MAX_NGRAMLIST_LEN]
    cdef np.uint8_t *codes_depTypes[MAX_NGRAMLIST_LEN]
    cdef int codelens_w0[MAX_NGRAMLIST_LEN]
    cdef int codelens_w1[MAX_NGRAMLIST_LEN]
    cdef int codelens_depTypes[MAX_NGRAMLIST_LEN]
    cdef np.uint32_t indexes_w0[MAX_NGRAMLIST_LEN]
#    cdef np.uint32_t indexes_w1[MAX_NGRAMLIST_LEN]

    cdef int ngram_list_len
    cdef REAL_t cnt

    cdef int i, j, k
    cdef long result = 0

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    ngram_list_len = <int>min(MAX_NGRAMLIST_LEN, len(ngrams))
    for i in range(ngram_list_len):
        counts[i]=1.0-(1.0/(<REAL_t> ngrams[i][3]+1))
        #counts[i]=1.0/(1.0+exp(-0.005*<REAL_t> ngrams[i][3]))	
        w0 = ngrams[i][0]
        if w0 is None:
            codelens_w0[i] = 0
        else:
            indexes_w0[i] = w0.index
            codelens_w0[i] = <int>len(w0.code)
#            codes_w0[i] = <np.uint8_t *>np.PyArray_DATA(w0.code)
#            points_w0[i] = <np.uint32_t *>np.PyArray_DATA(w0.point)
        w1 = ngrams[i][1]
        if w1 is None:
            codelens_w1[i] = 0
        else:
#            indexes_w1[i] = w1.index
            codelens_w1[i] = <int>len(w1.code)
            codes_w1[i] = <np.uint8_t *>np.PyArray_DATA(w1.code)
            points_w1[i] = <np.uint32_t *>np.PyArray_DATA(w1.point)
        depType = ngrams[i][2]
        if depType is None:
            codelens_depTypes[i] = 0
        else:
            codelens_depTypes[i] = <int>len(depType.code)
            codes_depTypes[i] = <np.uint8_t *>np.PyArray_DATA(depType.code)
            points_depTypes[i] = <np.uint32_t *>np.PyArray_DATA(depType.point)
        if w0 is not None and w1 is not None:
            result += 1

    # release GIL & train on the ngrams
    with nogil:
        for i in range(ngram_list_len):
            if codelens_w0[i] == 0 or codelens_w1[i] == 0:
                continue
#            j = i - window + reduced_windows[i]
#            if j < 0:
#                j = 0
#            k = i + window + 1 - reduced_windows[i]
#            if k > sentence_len:
#                k = sentence_len
#            for j in range(j, k):
#                if j == i or codelens[j] == 0:
#                    continue
            fast_sentence_ng(points_w1[i], codes_w1[i], codelens_w1[i], syn0, syn1, size, indexes_w0[i], points_depTypes[i], codes_depTypes[i], codelens_depTypes[i], _alpha*counts[i], work)
            #fast_sentence(points_w1[i], codes_w1[i], codelens_w1[i], syn0, syn1, size, indexes_w0[i], _alpha*counts[i], work)
            

    return result
    


def train_sentence(model, sentence, alpha, _work):
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *work
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            reduced_windows[i] = np.random.randint(window)
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                fast_sentence(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work)

    return result


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    global fast_sentence
    global fast_sentence_ng
    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        fast_sentence = fast_sentence0
        assert False #TODO: _ng version
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        fast_sentence = fast_sentence1
        fast_sentence_ng = fast_sentence1_ng
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        fast_sentence = fast_sentence2
        assert False #TODO: _ng version
        return 2

FAST_VERSION = init()  # initialize the module
