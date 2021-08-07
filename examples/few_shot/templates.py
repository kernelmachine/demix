from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import isabstract, signature
import re
from typing import Optional


TEMPLATES_REGISTRY = {}


def get_template_class_by_name(template_name):
    return TEMPLATES_REGISTRY[template_name.lower()]


def get_all_templates():
    return list(TEMPLATES_REGISTRY.keys())


@dataclass
class FewShotTemplate(ABC):

    task_description: Optional[str] = None

    @abstractmethod
    def encode(self, sample):
        raise NotImplementedError()
    
    def verbalize(self, sample, candidate):
        return candidate
    
    def postprocess(self, sample, candidate):
        return candidate
    
    def encode_correct_candidate(self, sample):
        return self.encode(sample).replace('<mask>', self.verbalize(sample, sample.correct_candidates[0]))  # Picking the first correct candidate
    
    @classmethod
    def from_kwargs(cls, **kwargs):
        """Allows instanciation from a kwargs dict even if it contains unused keys"""
        return cls(**{k: v for k, v in kwargs.items() if k in signature(cls).parameters})

    @classmethod
    def get_template_name(cls):
        [name] = re.match("(.+)Template", cls.__name__).groups()
        return name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if isabstract(cls):
            return
        template_name = cls.get_template_name()
        assert template_name not in TEMPLATES_REGISTRY, f"{template_name} template already registered!"
        TEMPLATES_REGISTRY[template_name] = cls


@dataclass
class COPATemplate(FewShotTemplate):

    capitalization: str = "correct"
    effect_conj: str = " so "
    cause_conj: str = " because "

    def encode(self, sample):
        premise = sample["premise"].rstrip()
        assert premise.endswith(".")
        premise = premise[:-1]

        if sample["question"] == "effect":
            conjunction = self.effect_conj
        elif sample["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError

        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()

        return prompt + '<mask>'

    def verbalize(self, sample, candidate):
        def capitalize(c):
            if self.capitalization == "correct":
                words = c.split(" ")
                if words[0] != "I":
                    words[0] = words[0].lower()
                return " ".join(words)
            elif self.capitalization == "bug":
                return c
            elif self.capitalization == "upper":
                return c.upper()
            elif self.capitalization == "lower":
                return c.lower()
            else:
                raise NotImplementedError

        return capitalize(sample[candidate])


@dataclass
class WiCTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        word = sample["word"]
        sentence1 = sample["sentence1"]
        sentence2 = sample["sentence2"]
        prompt = self.train_sep.join(
            [
                f"Sentence 1: {sentence1}",
                f"Sentence 2: {sentence2}",
                f"Question: Is the word '{word}' used in the same way in the two sentences above?",
                "Answer: <mask>",
            ]
        )
        return prompt
    
    def verbalize(self, sample, candidate):
        return {'false': 'False', 'true': 'True'}[candidate]


@dataclass
class BoolQTemplate(FewShotTemplate):
    train_sep: str = " "
    capitalization = "correct"

    def encode(self, sample):
        question = sample["question"]
        passage = sample["passage"]
        if self.capitalization == "correct":
            question = question.capitalize().strip("?") + "?"
        prompt = self.train_sep.join(
            [
                f"Paragraph: {passage}",
                f"Question: {question}",
                "Answer: <mask>",
            ]
        )
        return prompt
    
    def verbalize(self, sample, candidate):
        return {'false': 'False', 'true': 'True'}[candidate]


@dataclass
class BoolQNoPassageTemplate(BoolQTemplate):
    train_sep: str = " "

    def encode(self, sample):
        question = sample["question"]
        if self.capitalization == "correct":
            question = question.capitalize().strip("?") + "?"
        return f"Question: {question}{self.train_sep}Answer: <mask>"


@dataclass
class CBTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        question = sample["hypothesis"].capitalize().strip("?") + "?"
        prompt = self.train_sep.join(
            [
                f"Paragraph: {sample['premise']}",
                f"Question: {question} True, False, or Neither?",
                "Answer: <mask>",
            ]
        )
        return prompt
    
    def verbalize(self, sample, candidate):
        return {'entailment': 'True', 'contradiction': 'False', 'neutral': 'Neither'}[candidate]


@dataclass
class RTETemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        return self.train_sep.join([
            sample['premise'],
            f"question: {sample['hypothesis']}. True or False?",
            "answer: <mask>",
        ])
    
    def verbalize(self, sample, candidate):
        return {'entailment': 'True', 'not_entailment': 'False'}[candidate]


@dataclass
class WSCTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        tokens = sample['text'].split()
        txt1 = sample['target']['span1_text']
        txt2 = sample['target']['span2_text']
        idx1 = sample['target']['span1_index']
        idx2 = sample['target']['span2_index']
        # TODO This assertion fails because of incorrect tokenization
        # assert txt1.lower().startswith(tokens[idx1].lower()) and txt2.lower().startswith(tokens[idx2].lower())
        tokens[idx2] = '*' + txt2 + '*'  # Mark the pronoun in *bold*
        return self.train_sep.join([
            ' '.join(tokens),
            f'In the passage above, does the pronoun "*{txt2}*" refer to "{txt1}"? <mask>'
        ])
    
    def verbalize(self, sample, candidate):
        return {'false': 'No', 'true': 'Yes'}[candidate]


@dataclass
class ReCoRDTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        text = sample['passage']['text']
        text = text.replace('@highlight', '')
        query = sample['qas']['query']
        query = query.replace('@placeholder', '<mask>')
        return self.train_sep.join([text, query])
    
    def verbalize(self, sample, candidate):
        start, end = candidate
        txt = sample['passage']['text']
        return txt[start:end]


@dataclass
class MultiRCTemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        assert not sample.has_subproblems
        return self.train_sep.join([
            sample['text'],
            sample['question']['question'],
            f"- [<mask>] {sample['question']['answer']['text']}"
        ])
    
    def encode_correct_candidate(self, sample):
        assert sample.has_subproblems
        entries = [sample['text'], sample['question']['question']]
        for subproblem in sample.subproblems:
            gold = self.verbalize(subproblem, subproblem.correct_candidates[0])  # Picking the first correct candidate
            entries.append(f"- [{gold}] {subproblem['question']['answer']['text']}")
        return self.train_sep.join(entries)

    def verbalize(self, sample, candidate):
        return {'false': 'False', 'true': 'True'}[candidate]


@dataclass
class GPT3StyleNLITemplate(FewShotTemplate):
    train_sep: str = " "

    def encode(self, sample):
        prompt = self.train_sep.join(
            [
                f'Paragraph: {sample["sentence1"]}',
                f'Question: {sample["sentence2"]} True, False, or Neither?',
                "Answer: <mask>",
            ]
        )
        return prompt
    
    def verbalize(self, sample, candidate):
        return {'entailment': 'True', 'contradiction': 'False', 'neutral': 'Neither'}[candidate]


@dataclass
class PETStyleNLITemplate(FewShotTemplate):

    def encode(self, sample):
        sentence1 = sample["sentence1"].rstrip(".?!,;:'")
        sentence2 = sample["sentence2"]
        sentence2 = sentence2[0].lower() + sentence2[1:]  # Uncapitalize
        return f"{sentence1}? <mask>, {sentence2}"
    
    def verbalize(self, sample, candidate):
        return {'entailment': 'Yes', 'contradiction': 'No', 'neutral': 'Maybe'}[candidate]


@dataclass
class CycledLettersTemplate(FewShotTemplate):
    task_description: Optional[str] = "Please unscramble the letters into a word, and write that word:"

    def encode(self, sample):
        cycled_word = sample["cycled_word"]
        return f"{cycled_word} = <mask>"
    
    def postprocess(self, sample, candidate):
        return candidate.split(" ")[0].strip()


@dataclass
class SimplificationTemplate(FewShotTemplate):
    train_sep: str = " "
    separators: Optional[list] = field(default_factory=lambda: ["Paragraph:", "Question:", "Answer:"])

    def encode(self, sample):
        source = sample["source"]
        prompt = self.train_sep.join([
                f"Paragraph: {source}",
                "Question: Can you rephrase the above paragraph with simpler words to make it easier to read and understand by a child?",
                f"Answer: <mask>",
        ])
        return prompt
    
    def postprocess(self, sample, candidate):
        if self.separators is None:
            return candidate
        separators_regex = "(:?" + "|".join(self.separators) + ")"
        return re.split(separators_regex, candidate)[0].strip()
