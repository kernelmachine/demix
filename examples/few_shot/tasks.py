from abc import ABC, abstractmethod, abstractproperty
import copy
from dataclasses import dataclass, field
from functools import lru_cache
from inspect import isabstract, signature
import json
from typing import Optional, Union, Tuple
import re
from pathlib import Path

import numpy as np

from fairseq.data import data_utils
from examples.few_shot import templates
from examples.few_shot.metrics import FewShotMetric, AccuracyMetric, SariMetric


DATA_DIR = Path(__file__).resolve().parent / "data"
SUPERGLUE_DIR = DATA_DIR / "SuperGLUE"
FEW_SHOT_TASKS_REGISTRY = {}


def get_task_class_by_name(task_name):
    return FEW_SHOT_TASKS_REGISTRY[task_name.lower()]


def get_all_tasks():
    return list(FEW_SHOT_TASKS_REGISTRY.keys())


@lru_cache(maxsize=3)
def read_jsonl_file(filepath):
    json_objects = []
    with open(filepath, "r") as f:
        for line in f:
            json_objects.append(json.loads(line.rstrip()))
    return json_objects


def print_task(task_name):
    print(f"========== {task_name} examples ==========")
    task_class = get_task_class_by_name(task_name)
    task = task_class()
    print(task, "\n")
    for json_sample in task.eval_samples[:10]:
        print("-" * 80)
        print(task.format_priming_sample(json_sample))


def print_tasks(tasks=None):
    if tasks is None:
        tasks = get_all_tasks()
    for task_name in tasks:
        print_task(task_name)
        print("\n")


class FewShotSample(object):

    def __init__(self, data, candidates=None, correct_candidates=None, subproblems=None):
        self._data = data
        self._candidates = candidates
        self._correct_candidates = correct_candidates
        self._subproblems = subproblems if subproblems is not None else []
        if candidates is not None and correct_candidates is not None:
            assert all([correct_candidate in candidates for correct_candidate in correct_candidates])

    def __getitem__(self, key):
        return self._data[key]
    
    def __contains__(self, item):
        return item in self._data
    
    @property
    def candidates(self):
        return self._candidates
    
    @property
    def has_candidates(self):
        return self.candidates is not None and len(self.candidates) > 0
    
    @property
    def correct_candidates(self):
        return self._correct_candidates
    
    def is_correct(self, candidate):
        return candidate in self.correct_candidates
    
    @property
    def subproblems(self):
        return self._subproblems

    @property
    def has_subproblems(self):
        return len(self.subproblems) > 0


@dataclass
class FewShotTask(ABC):
    file_format = "jsonl"
    valid_file: Optional[Union[str, Path]] = None
    _train_samples = None
    _valid_samples = None
    _eval_samples = None
    n_eval_samples: Optional[int] = None
    metrics: Tuple[FewShotMetric] = (AccuracyMetric(), )

    @abstractproperty
    def train_file(self):
        pass

    @abstractproperty
    def eval_file(self):
        pass

    @abstractmethod
    def build_samples(self, parsed_data):
        pass
    
    @classmethod
    @abstractmethod
    def get_default_template_class(cls):
        raise NotImplementedError

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Allows instanciation from a kwargs dict even if it contains unused keys"""
        return cls(**{k: v for k, v in kwargs.items() if k in signature(cls).parameters})

    @classmethod
    def get_task_name(cls):
        [task_name] = re.match("(.+)Task", cls.__name__).groups()
        return task_name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if isabstract(cls):
            return
        task_name = cls.get_task_name()
        assert task_name not in FEW_SHOT_TASKS_REGISTRY, f"{task_name} task already registered!"
        FEW_SHOT_TASKS_REGISTRY[task_name] = cls

    def read_data(self, path):
        if path is None:
            return []
        elif self.file_format == "jsonl":
            return read_jsonl_file(path)
        else:
            raise NotImplementedError(f"Unsupported file format: {self.file_format}")

    @property
    def train_samples(self):
        if self._train_samples is None:
            self._train_samples = [sample for entry in self.read_data(self.train_file) for sample in self.build_samples(entry)]
        return self._train_samples

    @property
    def valid_samples(self):
        if self._valid_samples is None:
            self._valid_samples = [sample for entry in self.read_data(self.valid_file) for sample in self.build_samples(entry)]
        return self._valid_samples

    @property
    def eval_samples(self):
        if self._eval_samples is None:
            self._eval_samples = [sample for entry in self.read_data(self.eval_file) for sample in self.build_samples(entry)][:self.n_eval_samples]
        return self._eval_samples

    @property
    def has_candidates(self):
        # TODO Deciding based on the first eval sample
        return self.eval_samples[0].has_candidates

    def get_max_candidate_length(self, model):
        # TODO Return the length of the longest candidate for multiple choice tasks?
        raise NotImplementedError

    def get_random_subset(self, train_size, valid_size=0, uniform_sampling=False, seed=0):
        """Create a copy of this task with a random subset of the train/valid sets
        
        If the task doesn't have a validation set we create one from the original training set.
        """
        def random_subset(samples, k, candidate=None):
            samples = [sample for sample in samples if candidate is None or sample.correct_candidates[0] == candidate]
            idx = np.random.choice(len(samples), k, replace=False)
            return [samples[idx] for idx in idx]
        
        if uniform_sampling:
            assert self.has_candidates, "Cannot use uniform sampling with generative tasks"
            candidates = set()
            for samples in self.train_samples, self.valid_samples, self.eval_samples:
                for sample in samples:
                    assert not sample.has_subproblems, "Cannot use uniform sampling with tasks with subproblems"
                    assert len(sample.correct_candidates) == 1, "Cannot use uniform sampling with >1 correct candidates per instance"
                    candidates.update(sample.candidates)
            assert train_size % len(candidates) == 0, "train_size is not divisible by the number of classes"
            assert valid_size % len(candidates) == 0, "valid_size is not divisible by the number of classes"
        else:
            candidates = [None]
        with data_utils.numpy_seed(seed):
            subset = copy.deepcopy(self)
            subset._train_samples = []
            subset._valid_samples = []
            for candidate in candidates:
                if len(self.valid_samples) == 0:  # The original task doesn't have a valid set, so we will derive one from the training set
                    samples = random_subset(self.train_samples, (train_size + valid_size) // len(candidates), candidate)
                    subset._train_samples += samples[:train_size//len(candidates)]
                    subset._valid_samples += samples[train_size//len(candidates):]
                else:
                    subset._train_samples += random_subset(self.train_samples, train_size // len(candidates), candidate)
                    subset._valid_samples += random_subset(self.valid_samples, valid_size // len(candidates), candidate)
            # Shuffle the new train/valid sets
            subset._train_samples = random_subset(subset._train_samples, len(subset._train_samples))
            subset._valid_samples = random_subset(subset._valid_samples, len(subset._valid_samples))
            return subset


@dataclass
class COPATask(FewShotTask):
    eval_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "COPA/val.jsonl"
    train_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "COPA/train.jsonl"

    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            candidates = ["choice1", "choice2"],
            correct_candidates = ["choice1" if data["label"] == 0 else "choice2"],
        )]

    @classmethod
    def get_default_template_class(cls):
        return templates.COPATemplate


@dataclass
class WiCTask(FewShotTask):
    eval_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "WiC/val.jsonl"
    train_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "WiC/train.jsonl"

    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            candidates = ["false", "true"],
            correct_candidates = ["true" if data["label"] else "false"],
        )]
    
    @classmethod
    def get_default_template_class(cls):
        return templates.WiCTemplate


@dataclass
class BoolQTask(FewShotTask):
    eval_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "BoolQ/val.jsonl"
    train_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "BoolQ/train.jsonl"

    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            candidates = ["false", "true"],
            correct_candidates = ["true" if data["label"] else "false"],
        )]
    
    @classmethod
    def get_default_template_class(cls):
        return templates.BoolQTemplate


@dataclass
class CBTask(FewShotTask):
    eval_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "CB/val.jsonl"
    train_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "CB/train.jsonl"
    
    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            candidates = ["entailment", "contradiction", "neutral"],
            correct_candidates = [data["label"]],
        )]
    
    @classmethod
    def get_default_template_class(cls):
        return templates.CBTemplate


@dataclass
class RTETask(FewShotTask):
    eval_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "RTE/val.jsonl"
    train_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "RTE/train.jsonl"

    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            candidates = ["entailment", "not_entailment"],
            correct_candidates = [data["label"]],
        )]
    
    @classmethod
    def get_default_template_class(cls):
        return templates.RTETemplate


@dataclass
class WSCTask(FewShotTask):
    eval_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "WSC/val.jsonl"
    train_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "WSC/train.jsonl"

    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            candidates = ["false", "true"],
            correct_candidates = ["true" if data["label"] else "false"],
        )]
    
    @classmethod
    def get_default_template_class(cls):
        return templates.WSCTemplate


@dataclass
class ReCoRDTask(FewShotTask):
    eval_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "ReCoRD/val.jsonl"
    train_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "ReCoRD/train.jsonl"

    def build_samples(self, data):
        candidates = [(entity["start"], entity["end"] + 1) for entity in data["passage"]["entities"]]
        samples = []
        for qas in data["qas"]:
            correct_candidates = []
            for answer in qas["answers"]:
                candidate = (answer["start"], answer["end"] + 1)
                assert data["passage"]["text"][candidate[0]:candidate[1]] == answer["text"]
                correct_candidates.append(candidate)
            qas_data = dict(data)
            qas_data["qas"] = qas
            samples.append(FewShotSample(data=qas_data, candidates=candidates, correct_candidates=correct_candidates))
        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.ReCoRDTemplate


@dataclass
class MultiRCTask(FewShotTask):
    eval_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "MultiRC/val.jsonl"
    train_file: Optional[Union[str, Path]] = SUPERGLUE_DIR / "MultiRC/train.jsonl"

    def build_samples(self, data):
        data = data["passage"]
        samples = []
        for question in data["questions"]:
            sample_data = copy.deepcopy(data)
            del sample_data["questions"]
            sample_data["question"] = question
            subproblems = []
            for answer in question["answers"]:
                subproblem_data = copy.deepcopy(sample_data)
                del subproblem_data["question"]["answers"]
                subproblem_data["question"]["answer"] = answer
                subproblems.append(FewShotSample(
                    data = subproblem_data,
                    candidates = ["false", "true"],
                    correct_candidates = ["true" if answer["label"] else "false"],
                ))
            samples.append(FewShotSample(data=sample_data, candidates=["false", "true"], subproblems=subproblems))
        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.MultiRCTemplate


@dataclass
class AbstractXNLITask(FewShotTask):
    train_file: Optional[Union[str, Path]] = DATA_DIR / "XNLI-1.0/xnli.test.jsonl"  # HACK: we use the test set for few-shot for now
    eval_file: Optional[Union[str, Path]] = DATA_DIR / "XNLI-1.0/xnli.dev.jsonl"

    @abstractproperty
    def language(self):
        pass

    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            candidates = ["entailment", "contradiction", "neutral"],
            correct_candidates = [data["gold_label"]],
        )]
    
    @classmethod
    def get_default_template_class(cls):
        return templates.GPT3StyleNLITemplate
    
    def read_data(self, path):
        samples = super().read_data(path)
        return [sample for sample in samples if sample["language"] == self.language]


@dataclass
class EnXNLITask(AbstractXNLITask):
    language: str = "en"


@dataclass
class FrXNLITask(AbstractXNLITask):
    language: str = "fr"


@dataclass
class EsXNLITask(AbstractXNLITask):
    language: str = "es"


@dataclass
class ItXNLITask(AbstractXNLITask):
    language: str = "it"


@dataclass
class DeXNLITask(AbstractXNLITask):
    language: str = "de"


@dataclass
class CycledLettersTask(FewShotTask):
    train_file: Optional[Union[str, Path]] = DATA_DIR / "cycledletters/train.jsonl"
    eval_file: Optional[Union[str, Path]] = DATA_DIR / "cycledletters/val.jsonl"

    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            correct_candidates = [data["gold_word"]],
        )]

    def get_max_candidate_length(self, *args, **kwargs):
        return 10  # Should be large enough for all words
    
    @classmethod
    def get_default_template_class(cls):
        return templates.CycledLettersTemplate


@dataclass
class SimplificationTask(FewShotTask):
    train_file: Optional[Union[str, Path]] = DATA_DIR / "asset/test.jsonl"  # HACK: we use the test set for few-shot for now
    eval_file: Optional[Union[str, Path]] = DATA_DIR / "asset/valid.jsonl"
    metrics: Tuple[FewShotMetric] = (SariMetric(), )

    def build_samples(self, data):
        return [FewShotSample(
            data = data,
            correct_candidates = [data["references"][0]],  # We only take the first reference
        )]

    def get_max_candidate_length(self, *args, **kwargs):
        return 100  # Should be large enough for all samples

    @classmethod
    def get_default_template_class(cls):
        return templates.SimplificationTemplate
