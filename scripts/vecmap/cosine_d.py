import embeddings
import pdb
import numpy as np
import re
import sys
import time
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('--output_dir', default='./', help='the output dir')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')
    parser.add_argument('--out_name', type=str, default='cosine_d.npy', help='name of output')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    
    xp=np
    xp.random.seed(args.seed)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # compute cosine distance
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)
    src_size = x.shape[0]
    trg_size = z.shape[0]
    cosine_d = xp.zeros([src_size, trg_size])   

    cosine_d = x.dot(z.T)

    # Write cosine distance for S*T
    np.save(args.output_dir+args.out_name, cosine_d)
    

if __name__ == '__main__':
    main()
