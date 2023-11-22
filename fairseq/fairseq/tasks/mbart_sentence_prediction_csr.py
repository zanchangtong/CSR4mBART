# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import torch
from fairseq import utils
from fairseq.data import (
    LanguagePairDataset, 
    BaseWrapperDataset, 
    data_utils, 
    PrependTokenDataset, 
    RollDataset,
    ConcatSentencesDataset, 
    )
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from . import register_task
from .translation import TranslationTask

import os
import random
import numpy as np

logger = logging.getLogger(__name__)

def read_MUSE_bi_dict():
    dict_home = "path/to/bilingual_dict_MUSE"
    langs = ['ar', 'de', 'bg', 'el', 'es', 'fr', 'hi', 'ru', 'th', 'tr', 'vi', 'zh']
    dicts = {}
    for lang in langs:
        dicts['en-' + lang] = {}
        dict_path = os.path.join(dict_home, 'en-'+lang +'.txt')
        with open(dict_path, 'r', encoding = 'utf-8') as f:
            for w_pair in f:
                if lang == 'pt' or lang == 'ar' or lang == 'bg' or lang == 'el' or lang == 'hi' or lang == 'th' or lang == 'tr' or lang == 'vi':
                    w_en, w_tgt = w_pair.strip().split('\t')
                else:
                    w_en, w_tgt = w_pair.strip().split(' ')

                try:
                    dicts['en-' + lang][w_en].append(w_tgt)
                except KeyError:
                    dicts['en-' + lang][w_en] = []
                    dicts['en-' + lang][w_en].append(w_tgt)
    return dicts

class CSR_dataset(BaseWrapperDataset):
    def __init__(self, dataset, dictionary):
        super().__init__(dataset)
        self.dict_names = ['en-de', 'en-es', 'en-fr', 'en-it', 'en-pt']
        self.bi_dict=read_MUSE_bi_dict()
        self.pre_dict = dictionary
        
        from fairseq.data import encoders
        from omegaconf import OmegaConf
        bpe_cfg = OmegaConf.create(
            {'_name': 'sentencepiece', 'sentencepiece_model': '/path/to/mbart.cc25/sentence.bpe.model'}
            )
        self.bpe = encoders.build_bpe(bpe_cfg)
        tokenizer_cfg = OmegaConf.create({'_name': 'moses', 'moses_no_escape':'True'})
        self.tokenizer = encoders.build_tokenizer(tokenizer_cfg)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # ssl
        sample['target'] = sample['source']

        seperator_index = (sample['source'] == 2).nonzero(as_tuple=True)[0]
        sent1 = sample['source'][:seperator_index[0]+1]
        sent2 = sample['source'][seperator_index[0]+1:]
        sent1 = self.csr_bi_sent(sent1)
        sent1 = torch.cat((torch.tensor([0]), sent1[:-1]))
        sent2 = self.csr_bi_sent(sent2)
        sample['source'] = torch.cat((sent1, sent2))
        
        return sample
    
    def csr_bi_sent(self, tokens): 
        language_id = torch.tensor([tokens[-1]])
        
        tokens_str = self.pre_dict.string(tokens[:-1])
        sentence = self.bpe.decode(tokens_str)
        sentence = self.tokenizer.encode(sentence)
        sentence = sentence.split(' ')
        n = len(sentence)
        m = int( n * 0.35)
        a = [ i for i in range(n) ]
        random.shuffle(a)
        cur_dict = random.sample(self.dict_names, 1)[0]
        for i in a:
            is_title=False
            try:
                en_word = sentence[i]
                if en_word.istitle(): 
                    en_word = en_word.lower()
                    is_title = True
                sentence[i] = random.sample(self.bi_dict[cur_dict][en_word], 1)[0]
                if is_title:
                    sentence[i] = sentence[i].title()
                if sentence[i] == en_word:
                    continue
                m -= 1
                if m == 0:
                    break
            except KeyError:
                continue
        if not m == 0:
            print("one sentence not reach 0.35 ratio, replace: {} in {} ".format(m, n))
        sentence = ' '.join(sentence)
        sentence = self.tokenizer.decode(sentence)
        sentence = self.bpe.encode(sentence)
        cs_tokens = self.pre_dict.encode_line(sentence, add_if_not_exist=False)
        return torch.cat((cs_tokens, language_id))

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 5
        return n

    def size(self, index):
        n = self.dataset.size(index)
        if self.token is not None:
            n += 5
        return n

@register_task("mbart_sentence_prediction_csr")
class MBARTSentencePredictionCSRTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(",")
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol("[{}]".format(l))
            d.add_symbol("<mask>")
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(args.data, "input0", "dict.txt")
        )
        logger.info("[input] dictionary: {} types".format(len(src_dict)))
         
        tgt_dict = src_dict
        
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def get_path(key, split):
            return os.path.join(self.args.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )
        input1 = make_dataset("input1", self.source_dictionary)
        
        init_token = 0
        input0 = PrependTokenDataset(input0, init_token)

        if input1 is None:
            src_tokens = input0
        else:
            separator_token=None
            if separator_token is not None:
                input1 = PrependTokenDataset(input1, separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            '',
            'none',
            self.args.max_source_positions,
            self.args.seed,
        )
        # fix
        src_tokens = PrependTokenDataset(src_tokens, self.source_dictionary.index('[{}]'.format(self.args.append_srcid)))
        src_tokens = RollDataset(src_tokens, -1)
        
        tgt_dataset=src_tokens
        tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
        self.datasets[split] = LanguagePairDataset( 
            src_tokens,
            src_tokens.sizes,
            self.source_dictionary,
            tgt_dataset,
            tgt_dataset_sizes,
            self.source_dictionary,
            left_pad_source=True,
            left_pad_target=False,
            align_dataset=None,
            eos=self.source_dictionary.index('[{}]'.format(self.args.append_tgtid)),
            num_buckets=0,
            shuffle=True,
            pad_to_multiple=1,
        ) # use raw LanguagePairDataset class
        
        if split == 'train':
            self.datasets[split] = CSR_dataset(
                self.datasets[split], self.source_dictionary
            )

    def build_generator(self, models, args, **unused):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.args.append_tgtid)),
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator

            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                eos=self.tgt_dict.index("[{}]".format(self.args.append_tgtid)),
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        src_lang_id = self.source_dictionary.index("[{}]".format(self.args.append_srcid))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(
            source_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        ) # use raw LanguagePairDataset class
        return dataset
