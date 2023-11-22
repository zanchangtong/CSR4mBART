# -*-coding: UTF-8 -*-
import numpy as np
import argparse

def read(file):
    # read the vocabulary
    words = []
    for line in file:
        word, vec = line.split(' ', 1)
        words.append(word)
    print(len(words))
    return words

def try_index(list, token):
    try:  
        list.index(token)
    except ValueError:
        with open('error_token.txt', 'a', encoding='utf-8') as f:
            f.write(token+'\n')
        return -1
    else:
        return list.index(token)

parser = argparse.ArgumentParser(description='transform the index')
parser.add_argument('w2v_src_vocab', help='the word2vec source vocabulary')
parser.add_argument('w2v_tgt_vocab', help='the word2vec target vocabulary')
parser.add_argument('translation_src_vocab', help='the translation source vocabulary')
parser.add_argument('translation_tgt_vocab', help='the translation target vocabulary')
parser.add_argument('src2tgt', help='the Probabilistic translation lexicons from source to target')
parser.add_argument('lang_pair_dir', help='datadir wit lang_pair')
parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
args = parser.parse_args()

# construct the vocabularies of word2vec and translation

srcfile = open(args.w2v_src_vocab, encoding=args.encoding, errors='surrogateescape')
tgtfile = open(args.w2v_tgt_vocab, encoding=args.encoding, errors='surrogateescape')
w2v_src_vocab = read(srcfile)
w2v_tgt_vocab = read(tgtfile)

srcfile = open(args.translation_src_vocab, encoding=args.encoding, errors='surrogateescape') 
tgtfile = open(args.translation_tgt_vocab, encoding=args.encoding, errors='surrogateescape')
translation_src_vocab = read(srcfile)
translation_tgt_vocab = read(tgtfile)

# load src2tgt matrix
src2tgt = np.load(args.src2tgt)
 
# initial 
src_transformed_index=[] 
lexicons_transformed_index=np.zeros([sum(1 for line in open(args.w2v_src_vocab, encoding='utf-8')), 3])
S = 4 # for special token: 4 for BPE dictionary, 4 for mbart dictionary
for (i,token) in enumerate(w2v_src_vocab):
    index_translation = try_index(translation_src_vocab, token) + S 
    src_transformed_index.append(index_translation)

    lexicons_index = src2tgt[i].astype(int)
    for k in range(3):
        lexicon=w2v_tgt_vocab[lexicons_index[k]]
        lexicons_index_translation = try_index(translation_tgt_vocab,lexicon) + S
        lexicons_transformed_index[i][k]=lexicons_index_translation
        

for i in range(100):
    print(i)
    print(src_transformed_index[i])
    print(lexicons_transformed_index[i])

np.save(args.lang_pair_dir+'_spm_src.npy', np.array(src_transformed_index))
np.save(args.lang_pair_dir+'_spm_index.npy', np.array(lexicons_transformed_index))



