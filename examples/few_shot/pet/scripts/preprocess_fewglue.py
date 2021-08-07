import argparse
import json
import os
import string
from typing import Tuple, List, Union, Dict
from fairseq.data.encoders.gpt2_bpe import get_encoder

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]

MAX_LEN = 512

class InputExample(object):
    def __init__(self, text_a, text_b=None, label=None, logits=None):
        """
        Create a new InputExample.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits

class Processor(object):
    """Base processor to parse the data, bpe, truncate it and generate output files for each pattern"""
    def __init__(self, args):
        self.pattern_id = args.pattern_id
        self.mask = '<mask>'
        self.sep = '</s>'
        self.bpe = get_encoder(args.encoder_json, args.vocab_bpe)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True
    
    @staticmethod
    def _seq_length(parts: List[Tuple[List, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    
    def encode_part(self, part_str: str) -> str:
        return [part_str] if part_str == self.mask else self.bpe.encode(part_str)

    def encode_label(self, verbalizes: List) -> str:
        """bpe label string and aggregate verbalizes"""
        encoded_labels = []
        for verbalize in verbalizes:
            encoded_label = self.bpe.encode(verbalize)
            assert len(encoded_label) == 1, "current we only support single token label"
            encoded_labels.extend(encoded_label)
        return ",".join([str(label) for label in encoded_labels])

    
    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False):
        """
        convert examples to bped patterned strings
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        parts_a, parts_b = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(self.encode_part(x), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(self.encode_part(x), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b, max_length=MAX_LEN)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        input_ids = tokens_a if tokens_b is None else tokens_a + [self.sep] + tokens_b

        # pattern_id == -1 means traditional sentence classification
        if self.pattern_id != -1:
            all_labels = self.verbalize(example.label)
            encoded_labels = self.encode_label(all_labels)
        else:
            encoded_labels = example.label

        return " ".join([str(input_id) for input_id in input_ids]), encoded_labels



    def truncate(self, parts_a: List[Tuple[List, bool]], parts_b: List[Tuple[List, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)

        # hard code here for roberta sep and cls tokens
        if parts_b:
            total_len += 4
        else:
            total_len += 2

        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    def get_parts(self, example: InputExample) -> FilledPattern:
        pass

    def verbalize(self, label) -> List[str]:
        pass

class BoolQProcessor(Processor):
    """Processor for the BoolQ data set."""

    VERBALIZER_A = {
        "False": ["No"],
        "True": ["Yes"]
    }

    VERBALIZER_B = {
        "False": ["false"],
        "True": ["true"]
    }

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 2:
            return BoolQProcessor.VERBALIZER_A[label]
        else:
            return BoolQProcessor.VERBALIZER_B[label]

    def get_labels(self):
        return ["False", "True"]

    def get_parts(self, example: InputExample) -> FilledPattern:
        passage = self.shortenable(example.text_a)
        question = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [passage, '. Question: ', question, '? Answer: ', self.mask, '.'], []
        elif self.pattern_id == 1:
            return [passage, '. Based on the previous passage, ', question, '?', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Based on the following passage, ', question, '?', self.mask, '.', passage], []
        elif self.pattern_id == -1:
            return [passage], [question]
        else:
            raise Exception("invalid pattern")

    @staticmethod
    def create_examples(path: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = str(example_json['label']) if 'label' in example_json else None
                text_a = example_json['passage']
                text_b = example_json['question']
                example = InputExample(text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples

PROCESSOR_CLASS = {
    'boolq': BoolQProcessor,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="path for fewglue root directory")
    parser.add_argument("--output-dir", required=True, help="path for processed data folder")
    parser.add_argument("--task", required=True)
    parser.add_argument("--encoder-json", default="encoder.json", help="encoder json file for bpe")
    parser.add_argument("--vocab-bpe", default="vocab.bpe", help="vocab file for bpe")
    parser.add_argument("--pattern-id", type=int, default=0)
    args = parser.parse_args()

    # select and init processer
    processor = PROCESSOR_CLASS[args.task.lower()](args)
    
    # process training set
    set_types = ['train', 'valid']

    for set_type in set_types:
        input_path = os.path.join(args.input_dir, '{}.jsonl'.format(set_type))
        data_path = os.path.join(args.output_dir, '{}.input0.bpe'.format(set_type))
        label_path = os.path.join(args.output_dir, '{}.label.bpe'.format(set_type))
        examples = processor.create_examples(input_path)
        with open(data_path, 'w') as data_fout, open(label_path, 'w') as label_fout:
            for example in examples:
                data, label = processor.encode(example)
                data_fout.write(data + '\n')
                label_fout.write(label + '\n')


if __name__ == "__main__":
    main()