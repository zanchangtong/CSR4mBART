# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import warnings
from argparse import Namespace
from typing import List

import torch
from fairseq import metrics, search, tokenizer, utils
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


class FairseqTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @staticmethod
    def logging_outputs_can_be_summed(criterion) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `aggregate_logging_outputs`.
        Setting this to True will improves distributed training speed.
        """
        return criterion.logging_outputs_can_be_summed()

    def __init__(self, cfg: FairseqDataclass, **kwargs):
        self.cfg = cfg
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (omegaconf.DictConfig): parsed command-line arguments
        """
        return cls(cfg, **kwargs)

    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.cfg, "data", "")

    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        task_cfg: FairseqDataclass = None,
        **kwargs
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            combine (bool): combines a split segmented into pieces into one dataset
            task_cfg (FairseqDataclass): optional task configuration stored in the checkpoint that can be used
                                         to load datasets
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], FairseqDataset):
            raise TypeError("Datasets are expected to be of type FairseqDataset")
        return self.datasets[split]

    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions)
                )
            logger.warning(
                (
                    "{} samples have invalid sizes and will be skipped, "
                    "max_positions={}, first few sample ids={}"
                ).format(len(ignored), max_positions, ignored[:10])
            )
        return indices

    def can_reuse_epoch_itr(self, dataset):
        # We can reuse the epoch iterator across epochs as long as the dataset
        # hasn't disabled it. We default to ``False`` here, although in practice
        # this will be ``True`` for most datasets that inherit from
        # ``FairseqDataset`` due to the base implementation there.
        return getattr(dataset, "can_reuse_epoch_itr_across_epochs", False)
    def get_train_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ): 
        """
        padding for code-switch
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints; define_dir: data_utils.py
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater_CS,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            get_step=True,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints; define_dir: data_utils.py
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def build_model(self, cfg: FairseqDataclass):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            cfg (FairseqDataclass): configuration object

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models, quantization_utils

        model = models.build_model(cfg, self)
        if getattr(cfg, "tpu", False):
            model.prepare_for_tpu_()
        model = quantization_utils.quantize_model_scalar(model, cfg)
        return model

    def build_criterion(self, cfg: DictConfig):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            cfg (omegaconf.DictConfig): configration object

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions

        return criterions.build_criterion(cfg, self)

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
            else:
                seq_gen_cls = SequenceGenerator
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        return seq_gen_cls(
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
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def build_dataset_for_inference(
        self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass

    def begin_valid_epoch(self, epoch, model):
        """Hook function called before the start of each validation epoch."""
        pass

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        """[deprecated] Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "The aggregate_logging_outputs API is deprecated. "
            "Please use the reduce_metrics API instead."
        )
        with metrics.aggregate() as agg:
            self.reduce_metrics(logging_outputs, criterion)
            return agg.get_smoothed_values()

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs
        self_func = getattr(self, "aggregate_logging_outputs").__func__
        if self_func is not base_func:
            utils.deprecation_warning(
                "Tasks should implement the reduce_metrics API. "
                "Falling back to deprecated aggregate_logging_outputs API."
            )
            agg_logging_outputs = self.aggregate_logging_outputs(
                logging_outputs, criterion
            )
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    def build_tokenizer(self, args):
        """Build the pre-tokenizer for this task."""
        return encoders.build_tokenizer(args)

    def build_bpe(self, args):
        """Build the tokenizer for this task."""
        return encoders.build_bpe(args)
    
    # def get_train_batch_iterator(
    #     self,
    #     dataset,
    #     max_tokens=None,
    #     max_sentences=None,
    #     max_positions=None,
    #     ignore_invalid_inputs=False,
    #     required_batch_size_multiple=1,
    #     seed=1,
    #     num_shards=1,
    #     shard_id=0,
    #     num_workers=0,
    #     epoch=1,
    #     data_buffer_size=0,
    #     disable_iterator_cache=False,
    # ):

    #     can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
    #         dataset
    #     )
    #     if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
    #         logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
    #         return self.dataset_to_epoch_iter[dataset]

    #     assert isinstance(dataset, FairseqDataset)

    #     # initialize the dataset with the correct starting epoch
    #     dataset.set_epoch(epoch)

    #     # get indices ordered by example size
    #     with data_utils.numpy_seed(seed):
    #         indices = dataset.ordered_indices()

    #     # filter examples that are too large
    #     if max_positions is not None:
    #         indices = self.filter_indices_by_size(
    #             indices, dataset, max_positions, ignore_invalid_inputs
    #         )

    #     # TODO: 此处修改sampler和dataloader，对提取到的数据按照step进行code-switch。具体做法是：根据dataset创建sampler，对sampler中的数据还原，进行word级别的code-switch，二值化为网络输入，对句子依据长度排序处理。
    #     # samples: 'dict' object, {'id': tensor, 'nsentences': int, 'ntokens': int, 'net_input': {'src_tokens': tensor, 'src_lengths': tensor, 'prev_output_tokens': tensor}, 'target': tensor }
    #     ran_sampler = torch.utils.data.RandomSampler(dataset)  # get example = {"id": index, "source": src_item, "target": tgt_item,} batch开始之前有一个dummy，source和target长都为3
    #     dynamic_sampler = DynamicBatchSampler(ran_sampler,dataset.src_sizes ,dataset.num_tokens, num_buckets=200, max_size=1000, max_tokens=max_tokens, max_sentences=max_sentences)  # 自定义的batch sampler
    #     # 在dataloader中使用自定义的 sampler 和 batch sampler
    #     epoch_iter = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=dynamic_sampler)

    #     if can_reuse_epoch_itr:
    #         self.dataset_to_epoch_iter[dataset] = epoch_iter

    #     return epoch_iter


class LegacyFairseqTask(FairseqTask):
    def __init__(self, args: Namespace):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

    @classmethod
    def setup_task(cls, args: Namespace, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.args, "data", "")

    def build_model(self, args: Namespace):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models, quantization_utils

        model = models.build_model(args, self)
        if getattr(args, "tpu", False):
            model.prepare_for_tpu_()
        model = quantization_utils.quantize_model_scalar(model, args)
        return model

    def build_criterion(self, args: Namespace):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions

        return criterions.build_criterion(args, self)

# # 自定义sampler，执行code-switch操作构建batch。控制长度相似的句子在一个batch。
# class DynamicBatchSampler(torch.utils.data.Sampler):
#     def __init__(self, sampler, num_tokens_fn, token_length, num_buckets=100, min_size=0, max_size=1000,
#                  max_tokens=None, max_sentences=None, drop_last=False):
#         """

#         :param sampler:
#         :param num_tokens_fn: 根据idx返回样本的长度的函数
#         :param num_buckets: 利用桶原理将相似长度的样本放在一个batchsize中，桶的数量
#         :param min_size: 最小长度的样本， 小于这个值的样本会被过滤掉。 依据这个值来创建样桶
#         :param max_size: 最大长度的样本
#         :param max_sentences: batch_size, 但是这里可以通过max_sentences 和 max_tokens 共同控制最终的大小
#         """
#         super(DynamicBatchSampler, self).__init__(sampler)
#         self.sampler = sampler
#         self.num_tokens_fn = num_tokens_fn
#         self.num_buckets = num_buckets

#         self.min_size = min_size
#         self.max_size = max_size

#         assert max_size <= max_tokens, "max_size should be smaller than max tokens"
#         assert max_tokens is not None or max_sentences is not None, \
#             "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
#         self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
#         self.max_sentences = max_sentences if max_sentences is not None else float('Inf')
#         self.drop_last = drop_last

#     @property
#     def is_batch_full(self, num_tokens, batch):
#         if len(batch) == 0:
#             return False
#         if len(batch) == self.max_sentences:
#             return True
#         if num_tokens > self.max_tokens:
#             return True
#         return False

#     @property
#     def __iter__(self):

#         buckets = [[] for _ in range(self.num_buckets)]
#         sample_len = [0] * self.num_buckets 
        
#         for idx in self.sampler:
#             idx_length = self.num_tokens_fn(idx)
#             token_length = self.token_length(idx)
#             if not (self.min_size <= idx_length <= self.max_size):
#                 print("sentence at index {} of size {} exceeds max_tokens, the sentence is ignored".format(idx, idx_length))
#                 continue

#             index_buckets = math.floor((idx_length - self.min_size) / (self.max_size - self.min_size + 1)
#                                        * self.num_buckets)
#             sample_len[index_buckets] = max(sample_len[index_buckets], token_length)  # 是否

#             num_tokens = (len(buckets[index_buckets]) + 1) * sample_len[index_buckets]
#             if self.is_batch_full(num_tokens, buckets[index_buckets]):
#                 # yield this batch
#                 yield buckets[index_buckets]
#                 buckets[index_buckets] = []
#                 sample_len[index_buckets] = 0

#             buckets[index_buckets].append(idx)

#         # process left-over
#         leftover_batch = []
#         leftover_sample_len = 0
#         leftover = [idx for bucket in buckets for idx in bucket]
#         for idx in leftover:
#             idx_length = self.num_tokens_fn(idx)
#             leftover_sample_len = max(leftover_sample_len, idx_length)
#             num_tokens = (len(leftover_batch) + 1) * leftover_sample_len
#             if self.is_batch_full(num_tokens, leftover_batch):
#                 yield leftover_batch
#                 leftover_batch = []
#                 leftover_sample_len = 0
#             leftover_batch.append(idx)

#         if len(leftover_batch) > 0 and not self.drop_last:
#             yield leftover_batch

#     @property
#     def __len__(self):
#         # we do not know the exactly batch size, so do not call len(dataloader)
#         pass

# # 在sampler中增加缓存清除机制
# class RandomSampler(Sampler):
#     """Samples elements randomly. If without replacement, then sample from a shuffled dataset.
#     If with replacement, then user can specify ``num_samples`` to draw.

#     Arguments:
#         data_source (Dataset): dataset to sample from
#         num_samples (int): number of samples to draw, default=len(dataset)
#         replacement (bool): samples are drawn with replacement if ``True``, default=False
#     """

#     def __init__(self, data_source, replacement=False, num_samples=None):
#         self.data_source = data_source
#         self.replacement = replacement
#         self.num_samples = num_samples

#         if self.num_samples is not None and replacement is False:
#             raise ValueError("With replacement=False, num_samples should not be specified, "
#                              "since a random permute will be performed.")

#         if self.num_samples is None:
#             self.num_samples = len(self.data_source)

#         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
#             raise ValueError("num_samples should be a positive integeral "
#                              "value, but got num_samples={}".format(self.num_samples))
#         if not isinstance(self.replacement, bool):
#             raise ValueError("replacement should be a boolean value, but got "
#                              "replacement={}".format(self.replacement))

#     def __iter__(self):
#         # 使用排序之后的列表
#         return iter(self.data_source.ordered_indices)

#     def __len__(self):
#         return len(self.data_source)
