import sys
from word2vec import Word2Vec
import argparse
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Shrink a .bin model to a max_rank',usage='%(prog)s options source.bin target.bin')
parser.add_argument('-m','--max_rank', type=int, required=True, help='Max rank to use')
parser.add_argument('source_model',  help='Source model')
parser.add_argument('target_model',  help='Target model')
options = parser.parse_args()

w=Word2Vec.load_word2vec_format(options.source_model,binary=True)
w.save_word2vec_format(options.target_model,binary=True,max_rank=options.max_rank)

