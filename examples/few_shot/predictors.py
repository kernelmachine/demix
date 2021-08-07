from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from inspect import isabstract, signature
import numpy as np
import re
import torch

from fairseq.models import BaseFairseqModel

from examples.few_shot import tasks, templates
from examples.few_shot.models import convert_max_positions_to_int, run_with_adaptative_max_tokens

def print_r0(x, file=None):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(x, file=file)

PREDICTORS_REGISTRY = {}


def get_predictor_class_by_name(predictor_name):
    return PREDICTORS_REGISTRY[predictor_name.lower()]


def get_all_predictors():
    return list(PREDICTORS_REGISTRY.keys())


class ScoredCandidate(object):

    def __init__(self, candidate, score):
        self.candidate = candidate
        self.score = score


class Prediction(object):

    def __init__(self, sample, scored_candidates):
        self.sample = sample
        self.scored_candidates = scored_candidates

    @property
    def best_candidate(self):
        return max(self.scored_candidates, key = lambda x: x.score)


@dataclass
class FewShotPredictor(ABC):

    model: BaseFairseqModel
    task: tasks.FewShotTask
    
    @abstractmethod
    def predict(self, samples):
        pass

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Allows instanciation from a kwargs dict even if it contains unused keys"""
        return cls(**{k: v for k, v in kwargs.items() if k in signature(cls).parameters})

    @classmethod
    def get_predictor_name(cls):
        [name] = re.match("(.+)Predictor", cls.__name__).groups()
        return name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if isabstract(cls):
            return
        predictor_name = cls.get_predictor_name()
        assert predictor_name not in PREDICTORS_REGISTRY, f"{predictor_name} predictor already registered!"
        PREDICTORS_REGISTRY[predictor_name] = cls


@dataclass
class RandomPredictor(FewShotPredictor):
    
    def predict(self, samples):
        assert self.task.has_candidates
        samples = [subproblem for sample in samples for subproblem in (sample.subproblems if sample.has_subproblems else [sample])]  # Expand samples with subproblems (e.g., MultiRC)
        return [Prediction(sample=sample, scored_candidates=[ScoredCandidate(candidate=np.random.choice(sample.candidates), score=1.0)]) for sample in samples]


@dataclass
class MajorityClassPredictor(FewShotPredictor):

    def predict(self, samples):
        # We are picking the majority class in the test set
        assert self.task.has_candidates
        samples = [subproblem for sample in samples for subproblem in (sample.subproblems if sample.has_subproblems else [sample])]  # Expand samples with subproblems (e.g., MultiRC)
        correct_candidates = [candidate for sample in samples for candidate in sample.correct_candidates]
        majority_class = Counter(correct_candidates).most_common()[0][0]
        return [Prediction(sample=sample, scored_candidates=[ScoredCandidate(candidate=majority_class, score=1.0)]) for sample in samples]


@dataclass
class PromptingPredictor(FewShotPredictor):

    template: templates.FewShotTemplate
    train_sep: str = " "  # Slightly better performance with " " instead of "\n"
    
    def get_prompt(self, eval_sample, priming_samples):
        entries = []
        if self.template.task_description is not None:
            entries.append(self.template.task_description)
        for sample in priming_samples:
            entries.append(self.template.encode_correct_candidate(sample))
        entries.append(self.template.encode(eval_sample))
        return self.train_sep.join(entries)


@dataclass
class CLMPromptingPredictor(PromptingPredictor):

    beam_size: int = 1

    def truncate_prompt(self, prompt, max_tokens, debug=False):
        tokens = self.model.encode(prompt)
        tokens = tokens[tokens != self.model.task.dictionary.unk()]
        tokens = tokens[-max_tokens:]
        truncated_prompt = self.model.decode(tokens)
        if debug and truncated_prompt != prompt:
            nb_tokens = self.model.encode(prompt).numel()
            print(f"Truncated prompt to first {max_tokens} out of {nb_tokens} ({max_tokens/nb_tokens*100:.0f}%)")
        return truncated_prompt
    
    def score_candidates(self, samples):

        def get_common_prefix_length(tokens_list):
            common_prefix_length = 0
            for i in range(min([tokens.numel() for tokens in tokens_list])):
                if any(tokens[i] != tokens_list[0][i] for tokens in tokens_list):
                    break
                common_prefix_length += 1
            return common_prefix_length

        max_positions = convert_max_positions_to_int(self.model.max_positions)

        # Create all prompts
        prompts = []
        for sample in samples:
            prompt = self.get_prompt(sample, self.task.train_samples)
            candidates = [self.template.verbalize(sample, candidate) for candidate in sample.candidates]
            max_cand_length = max(self.model.encode(candidate).numel() for candidate in candidates)
            prompt = self.truncate_prompt(prompt, max_tokens=max_positions - max_cand_length - 1)
            prompts.extend([prompt.replace('<mask>', candidate) for candidate in candidates])
        print_r0(f"iterated over {len(samples)} samples")

        # Predict
        unique_prompts = list(set(prompts))  # Deduplicate for efficiency
        unique_hypotheses = run_with_adaptative_max_tokens(self.model, self.model.score, sentences=unique_prompts)
        prompt2hypothesis = {prompt: hypothesis for prompt, hypothesis in zip(unique_prompts, unique_hypotheses)}
        hypotheses = [prompt2hypothesis[prompt] for prompt in prompts]
        print_r0(f"predicted {len(unique_prompts)} unique_prompts")
        # Score the results
        predictions = []
        for sample in samples:
            hypotheses_batch = [hypotheses.pop(0) for _ in sample.candidates]
            common_prefix_length = get_common_prefix_length([hypo["tokens"] for hypo in hypotheses_batch])
            scored_candidates = []
            for candidate, hypothesis in zip(sample.candidates, hypotheses_batch):
                scored_candidates.append(ScoredCandidate(
                    candidate=candidate,
                    score = hypothesis["positional_scores"][common_prefix_length:].mean(),
                ))
            predictions.append(Prediction(sample=sample, scored_candidates=scored_candidates))
        print_r0(f"predictions calculated for {len(samples)} samples")
        return predictions

    def generate(self, samples):
        # TODO: Does not work for seq2seq models (teacher forcing not used for decoder in seq2seq_lm task)
        max_positions = convert_max_positions_to_int(self.model.max_positions)

        # Create all prompts
        prompts = []
        for eval_sample in samples:
            prompt = self.get_prompt(eval_sample, self.task.train_samples)
            assert prompt.endswith('<mask>')
            prompt = prompt[:-6].rstrip()  # Remove <mask>
            prompt = self.truncate_prompt(prompt, max_tokens=max_positions - self.task.get_max_candidate_length())
            prompts.append(prompt)

        # Predict
        generations = run_with_adaptative_max_tokens(self.model, self.model.sample, sentences=prompts, beam=self.beam_size, max_len_b=max_positions)

        # Postprocess
        predictions = []
        for generation, prompt, sample in zip(generations, prompts, samples):
            assert generation[:len(prompt)] == prompt, f"{generation}\n\n{prompt}"  # TODO: This does not pass for seq2seq models yet
            generation = generation[len(prompt):]  # Strip the prompt to keep only actual generated text
            generation = self.template.postprocess(sample, generation)
            predictions.append(Prediction(sample=sample, scored_candidates=[ScoredCandidate(candidate=generation, score=1.0)]))
        return predictions

    def predict(self, samples):
        samples = [subproblem for sample in samples for subproblem in (sample.subproblems if sample.has_subproblems else [sample])]  # Expand samples with subproblems (e.g., MultiRC)
        if self.task.has_candidates:  # For multiple choice tasks we score all candidates
            return self.score_candidates(samples)
        else:
            return self.generate(samples)
