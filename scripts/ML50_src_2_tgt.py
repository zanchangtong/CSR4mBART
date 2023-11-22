import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Build the bilingual dictionary')
parser.add_argument('src_input', help='the input source embeddings')
parser.add_argument('output_dir', help='the output dir')
parser.add_argument('language_pair', help='language_pair')
args = parser.parse_args()

print('top-3 select:', args.src_input)
CS_dis = np.load(args.src_input)
print(CS_dis.shape)
src_2_tgt=np.zeros([CS_dis.shape[0], 3])
for j in range(CS_dis.shape[0]):
    src_2_tgt[j] = np.argsort(CS_dis[j][:])[:3]
np.save(args.output_dir+args.language_pair+'.npy', src_2_tgt)
print('top-3 selected shape:', src_2_tgt.shape)
