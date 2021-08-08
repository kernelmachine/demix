# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
import ast
from tqdm import tqdm, trange

from torch.utils.data import Subset
import numpy as np
import torch
from collections import defaultdict
from fairseq import utils
from fairseq.data import (
    ConcatDataset,
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    MonodomainDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    ResamplingDataset,
    MultiCorpusSampledDataset,
    SortDataset,
    StripTokenDataset,
    TokenBlockDataset,
    SubsampleDataset,
    TruncatedDictionary,
    data_utils,
    SubsetDataset
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.data.cluster_dataset import ClusterDataset,SingleClusterDataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II
from torch.utils.data import DataLoader

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.decomposition import TruncatedSVD

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class MultidomainLanguageModelingConfig(FairseqDataclass):
    # TODO common var add to parent
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})

    recluster_data: bool = field(default=False, metadata={"help": "recluster data"})

    add_domain_token: bool = field(
        default=False, metadata={"help": "prepend domain token <domain>"}
    )

    force_domain_token: Optional[str] = field(
        default=None, metadata={"help": "force specific domain token <domain>"}
    )

    domain_parallel: bool = field(
        default=False, metadata={"help": "if set, perform domain_parallel training"}
    )

    gpu_mappings: Optional[str] = field(
        default=None, metadata={"help": "dictionary of GPUs indexes to domain indexes"}
    )

    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False, metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False, metadata={"help": "boolean to pad to fixed batch size"},
    )

    multidomain_sampling_alpha: Optional[float] = field(
        default=1.0, metadata={"help": "smoothing alpha for sample rations across multiple datasets"}
    )

    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )


    train_domains: str = field(
        default="",
        metadata={
            "help": "comma-separated list of domains (default: all directories in data path)"
        },
    )
    eval_domains: str = field(
        default="",
        metadata={
            "help": "comma-separated list of domains (default: all directories in data path)"
        },
    )

    unbalanced: bool = field(
        default=False,
        metadata={
            "help": "comma-separated list of domains (default: all directories in data path)"
        },
    )

    original_domains: str = field(
        default="",
        metadata={
            "help": "comma-separated list of domains (default: all directories in data path)"
        },
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    train_subset: str = II("common.train_subset")
    valid_subset: str = II("common.valid_subset")


def domain_token(domain):
    return f"<{domain}>"


@register_task("multidomain_language_modeling", dataclass=MultidomainLanguageModelingConfig)
class MultidomainLanguageModelingTask(LegacyFairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary

        if targets is None:
            targets = ["future"]
        self.targets = targets

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            if args.add_domain_token:
                if args.original_domains:
                    for domain in args.original_domains.split(","):
                        dictionary.add_symbol(domain_token(domain))
                else:
                    train_domains, eval_domains, _ = cls._get_domains(args)
                    for domain in train_domains:
                        dictionary.add_symbol(domain_token(domain))
                    if train_domains != eval_domains:
                        for domain in eval_domains:
                            dictionary.add_symbol(domain_token(domain))

            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )
        return (dictionary, output_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)

        # upgrade old checkpoints
        if hasattr(args, "exclude_self_target"):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)

        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by domains. This helps low resource
        domains by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multidomain_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    @classmethod
    def _get_domains(cls, args, epoch=1):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        domains = sorted(
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        )

        if args.train_domains:
            keep_domains = set(args.train_domains.split(','))
            train_domains = [domain for domain in domains if domain in keep_domains]
            # assert len(train_domains) == len(keep_domains)
        else:
            train_domains = []
        if args.eval_domains:
            keep_domains = set(args.eval_domains.split(','))
            eval_domains = [domain for domain in domains if domain in keep_domains]
            # assert len(eval_domains) == len(keep_domains)
        else:
            eval_domains = []
        if not args.train_domains and not args.eval_domains:
            train_domains = domains
            eval_domains = domains
        return train_domains, eval_domains, data_path

    def add_domain_token(self, token):
        logger.info(f"adding domain token {token} to dictionary")
        self.dictionary.add_symbol(domain_token(token))

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        train_domains, eval_domains, data_path = MultidomainLanguageModelingTask._get_domains(self.args, epoch)
        logger.info("Training on {0} domains: {1}".format(len(train_domains), train_domains))
        tokens_per_sample = self.args.tokens_per_sample - int(self.args.add_domain_token)

        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = self.args.batch_size_valid if 'valid' in split else self.args.batch_size

        sorted(train_domains)
        sorted(eval_domains)
        domain_datasets = []
        if torch.distributed.is_initialized():
            if not self.args.unbalanced:
                if split in self.args.train_subset.split(','):
                    gpus  = range(torch.distributed.get_world_size())
                    if len(gpus) >= 8:
                        num_gpus_per_domain = torch.distributed.get_world_size() // 8
                    else:
                        num_gpus_per_domain = 1
                    gpu_mappings = [list(gpus[n:n+num_gpus_per_domain]) for n in range(0, len(gpus), num_gpus_per_domain)]
                    mappings = {}
                    for ix, gpus in enumerate(gpu_mappings):
                        for gpu in gpus:
                            mappings[gpu] = ix
                    domain_id = mappings[torch.distributed.get_rank(torch.distributed.group.WORLD)]
                    train_domain = train_domains[domain_id]
                    domains = [(domain_id, train_domain)]
                else:
                    domains = enumerate(eval_domains)
            else:
                domains = enumerate(train_domains)
        # elif torch.distributed.is_initialized() and self.args.domain_parallel:
        #     gpus = range(torch.distributed.get_world_size())
        #     if len(gpus) >= 8:
        #         num_gpus_per_domain = torch.distributed.get_world_size() // 8
        #     else:
        #         num_gpus_per_domain = 1
        #     gpu_mappings = [list(gpus[n:n+num_gpus_per_domain]) for n in range(0, len(gpus), num_gpus_per_domain)]
        #     mappings = {}
        #     for ix, gpus in enumerate(gpu_mappings):
        #         for gpu in gpus:
        #             mappings[gpu] = ix

        #     domain_id = mappings[torch.distributed.get_rank(torch.distributed.group.WORLD)]
        #     train_domain = train_domains[domain_id]
        #     domains = [(domain_id, train_domain)] if split in self.args.train_subset.split(',') else enumerate(eval_domains)
        else:
            domains = [(0, eval_domains[0])]

        # if split in self.args.train_subset.split(',') in len(split.split('_')) > 1 and split.split('_')[1] != train_domain:
            # return

        # def dedup(seq):
        #     seen = set()
        #     seen_add = seen.add
        #     return [x for x in seq if not (x in seen or seen_add(x))]

        # unique_domains = dedup(train_domains + eval_domains)


        domain_datasets = []

        for domain_id, domain in domains:
            split_path = os.path.join(data_path, domain, split)
            dataset = data_utils.load_indexed_dataset(
                split_path, self.dictionary, self.args.dataset_impl, combine=combine
            )


            if dataset is None:
                continue

            dataset = maybe_shorten_dataset(
                dataset,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                tokens_per_sample,
                self.args.seed,
            )


            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode=self.args.sample_break_mode,
                include_targets=True,
            )

            add_eos_for_other_targets = (
                self.args.sample_break_mode is not None
                and self.args.sample_break_mode != "none"
            )
            src_domain_idx, tgt_domain_idx = domain_id, domain_id
            src_domain_token, tgt_domain_token = None, None
            if self.args.add_domain_token:
                if self.args.force_domain_token:
                    src_domain_token = self.dictionary.index(domain_token(self.args.force_domain_token))
                    tgt_domain_token = self.output_dictionary.index(domain_token(self.args.force_domain_token))
                else:
                    src_domain_token = self.dictionary.index(domain_token(domain))
                    tgt_domain_token = self.output_dictionary.index(domain_token(domain))

            domain_dataset = MonodomainDataset(
                    dataset=dataset,
                    sizes=dataset.sizes,
                    src_vocab=self.dictionary,
                    tgt_vocab=self.output_dictionary,
                    add_eos_for_other_targets=add_eos_for_other_targets,
                    shuffle=True,
                    targets=self.targets,
                    fixed_pad_length=fixed_pad_length,
                    pad_to_bsz=pad_to_bsz,
                    add_domain_token=self.args.add_domain_token,
                    src_domain_idx=src_domain_idx,
                    tgt_domain_idx=tgt_domain_idx,
                    src_domain_token=src_domain_token,
                    tgt_domain_token=tgt_domain_token
                )

            if self.args.recluster_data and split in self.args.train_subset.split(','):
                kmeans = MiniBatchKMeans(n_clusters=8)
                svd = TruncatedSVD(n_components=64)
                vectorizer = CountVectorizer(vocabulary=self.dictionary.symbols,
                                            tokenizer=lambda x: x,
                                            analyzer = lambda x:x,
                                            preprocessor=lambda x: x)

                subsets = []
                pretrain_dataset = SubsampleDataset(domain_dataset, size_ratio=0.001)
                text = []
                for idx in trange(len(pretrain_dataset), disable=torch.distributed.get_rank() !=0):
                    tt = self.dictionary.string(pretrain_dataset.__getitem__(idx)['source'])
                    text.extend(tt.split('\n'))
                vec = vectorizer.fit_transform(tqdm(text, disable=torch.distributed.get_rank() !=0))
                vec = svd.fit_transform(vec)
                kmeans.partial_fit(vec)

            domain_datasets.append(domain_dataset)

        if self.args.recluster_data:
            domain_datasets = [ClusterDataset(domain_dataset, vectorizer, svd, kmeans) for domain_dataset in domain_datasets]

        dataset_lengths = np.array(
            [len(d) for d in domain_datasets],
            dtype=float,
        )

        logger.info(
            "loaded total {} blocks for all domains".format(
                dataset_lengths.sum(),
            )
        )



        if split in self.args.train_subset.split(','):
            from fairseq import pdb; pdb.set_trace()
            if self.args.recluster_data:
                clusters = defaultdict(list)

                for dataset in tqdm(domain_datasets, disable=torch.distributed.get_rank() != 0):
                    indices = defaultdict(list)
                    loader = DataLoader(dataset, batch_size=16, num_workers=16)
                    for item in trange(loader, disable=torch.distributed.get_rank() != 0):
                        indices[item['cluster']].extend(item['id'])
                    for cluster, idxs in indices.items():
                        clusters[cluster].append(SubsetDataset(dataset, idxs))

                domain_datasets_ = []
                for cluster, ds in domain_datasets.items():
                    domain_datasets_.append(ConcatDataset(ds))
                domain_datasets = domain_datasets_
            if not self.args.unbalanced:
                ds = OrderedDict()
                for i, d in enumerate(domain_datasets):
                    ds[i] = d
                if self.args.gpu_mappings is not None:
                    gpu_mappings = ast.literal_eval(self.args.gpu_mappings)
                else:
                    gpu_mappings = None
                lens = torch.tensor([len(d) for d in domain_datasets]).cuda()
                gather_lens = [torch.ones_like(lens[0]).cuda() for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gather_lens, lens)
                gather_lens = [x.item() for x in gather_lens]
                for ix, item in enumerate(gather_lens):
                    logger.info(
                    "loaded total {} blocks on GPU {}".format(
                    item, ix
                )
            )
                dataset = MultiCorpusSampledDataset(ds, gather_lens, gpu_mappings=gpu_mappings)
            else:
                dataset = ConcatDataset(domain_datasets)
        else:

            ds = []
            size_ratio = np.array([0.1] * len(domain_datasets))
            for i, d in enumerate(domain_datasets):
                d = ResamplingDataset(
                    domain_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                ds.append(d)
            dataset = ConcatDataset(ds)


            # domain_splits = [split]
            # for domain_id, domain_dataset in enumerate(domain_datasets):
            #     split_name = split + "_" + eval_domains[domain_id]
            #     domain_splits.append(split_name)
            #     self.datasets[split_name] = domain_dataset
            # from fairseq import pdb; pdb.set_trace()
            # # [TODO]: This is hacky for now to print validation ppl for each
            # # language individually. Maybe need task API changes to allow it
            # # in more generic ways.

        # if self.args.domain_parallel:
        self.datasets[split] = dataset
        # else:
        #     with data_utils.numpy_seed(self.args.seed + epoch):
        #         shuffle = np.random.permutation(len(dataset))
        #     self.datasets[split] =  SortDataset(
        #     dataset,
        #     sort_order=[
        #         shuffle,
        #         dataset.sizes,
        #     ],
        # )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        ).next_epoch_itr(shuffle=False)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )
