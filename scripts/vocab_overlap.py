import sys
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import json
from typing import List
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
import pandas as pd


sns.set(context="paper", style="white", font_scale=1.4) 

def load_data(data_path: str, sample: int=None) -> List[str]:
    examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}", disable=sample is None) as f:
        for line in f:
            if sample:
                if len(examples) > sample:
                    break
            line = line.strip()
            if line:
                if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                    example = json.loads(line)
                else:
                    example = {"text": line}
                text = example['text']
                if sample:
                    if np.random.binomial(1, 0.5):
                        examples.append(text)
                else:
                    examples.append(text)
    if sample:
        examples = np.random.choice(examples, size=sample)
    return examples

def load_vocab(file, sample):
    if 'reviews' in file or '1b' in file or 'reddit' in file:
        sample = sample * 5
    text = load_data(file, sample)
    count_vectorizer = CountVectorizer(stop_words="english", min_df=3, ngram_range=(2,2))
    pbar = tqdm(text)
    pbar.set_description(file)
    count_vectorizer.fit(pbar)
    vocab = set(count_vectorizer.vocabulary_.keys())
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="files containing tokenized text from each domain", required=True)
    parser.add_argument("--output_plot_file", help="path to save heatmap", required=True)
    parser.add_argument("--output_data_file", help="path to save heatmap data", required=True)
    parser.add_argument("--sample", type=int, help="sample documents", required=False)

    args = parser.parse_args()
    files = args.files
    sample = args.sample
    vocabs = {}
    for file in files:
        sample  = args.sample
        if 'med' in file:
            key = 'BM'
        elif 'review' in file:
            key = 'RV'
        elif 'cs' in file:
            key = 'CS'
        elif 'realnews' in file:
            key = 'RN'
        elif 'webtext' in file:
            key = 'WT'
        elif 'reddit' in file:
            key = 'RD'
        elif 'legal' in file:
            key = 'LG' 
        elif '1b' in file:
            key = '1B'
        vocabs[key] = load_vocab(file, args.sample)
    
    file_pairs = itertools.combinations(list(vocabs.keys()), 2)
    
    overlaps = {}
    for x, y in file_pairs:
        intersection = vocabs[x] & vocabs[y]
        union = (vocabs[x] | vocabs[y])
        overlaps[x + "_" + y] = len(intersection) / len(union)
    
    data = []



z = {}
for key in overlaps.keys():
    file_1, file_2 = key.split('_')
    if not z.get(file_1):
        z[file_1] = {}
    z[file_1][file_2] = overlaps[key]
    if not z.get(file_2):
        z[file_2] = {}
    z[file_2][file_1] = overlaps[key]

labels = ["1B", "CS", "LG","BM", "WT","RN", "RD", "RV"]

for ix, key in enumerate(labels):
    items = []
    for subkey in labels:
        if not z[key].get(subkey):
            items.append(1.0)
        else:
            items.append(z[key][subkey])
    data.append(items)
data = np.array(data) * 100
np.save(args.output_data_file, data)


fig, ax = plt.subplots(1,1,figsize=(8,8))
sns.heatmap(data, cmap="Blues", vmin=30, xticklabels=labels, annot=True, fmt=".1f", cbar=False, yticklabels=labels, ax=ax)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(args.output_plot_file, dpi=300)
