# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import random

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    BaseWrapperDataset,
    ColorizeDataset,
    ConcatDataset,
    MonolingualDataset,
    PrependDataset,
    ReplaceDataset,
    SubsampleDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask


class ds_name_getter:
    def __init__(self, offset, generic_ds_name_chance, dictionary):
        self.offset = offset
        self.generic_ds_name_chance = generic_ds_name_chance
        self.dictionary = dictionary

    def __call__(self, dataset, index):
        if (
            self.generic_ds_name_chance > 0
            and np.random.rand() <= self.generic_ds_name_chance
        ):
            name = "generic"
        else:
            name = dataset.attr("name", index)
        assert name is not None
        return self.dictionary.indices[name] + self.offset


@register_task("tagged_language_modeling")
class TaggedLanguageModelingTask(LanguageModelingTask):
    """
    Like the language modeling task, but prepends tags to each sample
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        LanguageModelingTask.add_args(parser)
        parser.add_argument(
            "--multiple-datasets",
            action="store_true",
            help="if set, treats paths in data as separate datasets to be combined, "
            "rather than as splits of a single dataset",
        )
        parser.add_argument(
            "--prepend-ds-name",
            action="store_true",
            help="if set and multiple-datasets is also set, prepends the name of the ds instead of "
            "bos/eos token",
        )
        parser.add_argument(
            "--colorize-ds-name",
            action="store_true",
            help="if set and multiple-datasets is also set, adds an embedding for a specific dataset to source",
        )
        parser.add_argument(
            "--generic-ds-name-chance",
            type=float,
            metavar="P",
            default=0,
            help='if multiple datasets is used, sets the prepended ds name to "generic" '
            "this percentage of time",
        )
        parser.add_argument(
            "--subsample-splits",
            type=str,
            metavar="SPLITS",
            default="valid",
            help="if multiple datasets is used, subsamples specified split(colon separated) to "
            "the size of the smallest split",
        )

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)
        self.subsample_splits = (
            set()
            if args.subsample_splits is None
            else set(args.subsample_splits.split(":"))
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        if self.args.multiple_datasets:
            if len(paths) == 1:
                paths = [os.path.join(paths[0], p) for p in next(os.walk(paths[0]))[1]]
            datasets = [
                ShardedDataset(
                    self.dictionary,
                    self.args.dataset_impl,
                    path,
                    split,
                    epoch - 1,
                    combine=combine,
                )
                for path in paths
            ]

            ds_names = [ds.name for ds in datasets]

            if split in self.subsample_splits:
                sizes = [sum(d.sizes) for d in datasets]
                min_sz = min(sizes)
                ratios = [min_sz / sz for sz in sizes]
                datasets = [
                    SubsampleDataset(d, r) if r < 1 else d
                    for d, r in zip(datasets, ratios)
                ]

            dataset = ConcatDataset(datasets)
        else:
            data_path = paths[(epoch - 1) % len(paths)]
            split_path = os.path.join(data_path, split)

            dataset = data_utils.load_indexed_dataset(
                split_path, self.dictionary, self.args.dataset_impl, combine=combine
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )
            ds_names = [None]

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
        )

        if self.args.prepend_ds_name:
            dataset = PrependDataset(
                dataset,
                prepend_getter=ds_name_getter(
                    offset=0,
                    generic_ds_name_chance=self.args.generic_ds_name_chance,
                    dictionary=self.dictionary,
                ),
                ensure_first_token_is=self.dictionary.eos(),
            )

        dataset = ReplaceDataset(
            dataset,
            replace_map={self.dictionary.eos(): self.dictionary.indices["\\n"]},
            offsets=[1, -1],
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )

        dataset = MonolingualDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
        )

        if self.args.colorize_ds_name:
            ds_names.append("generic")
            min_ds = min(self.dictionary.indices[n] for n in ds_names)
            dataset = ColorizeDataset(
                dataset,
                color_getter=ds_name_getter(
                    offset=-min_ds,
                    generic_ds_name_chance=self.args.generic_ds_name_chance,
                    dictionary=self.dictionary,
                ),
            )

        self.datasets[split] = dataset

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            tag = None
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample["net_input"]["src_tokens"]
                tag = (
                    self.dictionary.indices[sample["tag"]]
                    if self.args.prepend_ds_name
                    else None
                )
            if self.args.colorize_ds_name:
                color_ind = (
                    self.dictionary.indices[sample["tag"]]
                    - self.dictionary.indices["wikipedia_gpt2"]
                )
                sample["decoder_input"] = {
                    "colors": torch.tensor([color_ind], dtype=torch.long)
                }
                if sample["net_input"]["src_tokens"].is_cuda:
                    sample["decoder_input"]["colors"] = sample["decoder_input"][
                        "colors"
                    ].cuda()

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=tag
            )


class ShardedDataset(BaseWrapperDataset):
    """Loads a dataset which has been sharded into multiple files.

    Each shard is only loaded for each specific epoch.
    """

    def __init__(
        self,
        dictionary,
        dataset_impl: str,
        path: str,
        split: str,
        epoch: int,
        name: str = None,
        combine: bool = False,
        seed: int = 0,
    ):
        self._name = name if name is not None else os.path.basename(path)
        num_shards = 0
        for i in itertools.count():
            if not os.path.exists(os.path.join(path, "shard" + str(i))):
                break
            num_shards += 1

        if num_shards > 0 and split == "train":
            random.seed(seed ^ epoch)
            shard = random.randint(0, num_shards - 1)
            split_path = os.path.join(path, "shard" + str(shard), split)
        else:
            split_path = os.path.join(path, split)
            if os.path.isdir(split_path):
                split_path = os.path.join(split_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, dictionary, dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        super().__init__(dataset)

    @property
    def name(self):
        return self._name
