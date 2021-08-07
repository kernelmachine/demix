import argparse
import json
import logging
import os
import sys
from typing import Iterator, Iterable, TypeVar, List
from itertools import islice
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import submitit
import numpy as np
import pandas as pd
import time
from pathlib import Path

A = TypeVar("A")


def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break

def get_json_data(input_file, predictor=None, ids_already_done=[]):
    if ids_already_done:
        set_ids_already_done = {x : 1 for x in ids_already_done}
    else:
        set_ids_already_done = {}
    if input_file == "-":
        for line in sys.stdin:
            if not line.isspace():
                if predictor:
                    res = predictor.load_line(line)
                    if not set_ids_already_done.get(res['id']):
                        yield res
                else:
                    res = json.loads(line)
                    if not set_ids_already_done.get(res['id']):
                        yield res
    else:
        with open(input_file, "r") as file_input:
            for line in tqdm(file_input):
                if not line.isspace():
                    if predictor:
                        res = predictor.load_line(line)
                        if not set_ids_already_done.get(res['id']):
                            yield res
                    else:
                        res = json.loads(line)
                        if not set_ids_already_done.get(res['id']):
                            yield res

def predict_json(predictor, batch_data):
    if len(batch_data) == 1:
        results = [predictor.predict_json(batch_data[0])]
    else:
        results = predictor.predict_batch_json(batch_data)
    for output in results:
        yield output

def extract(input_df, output_file, model_path, batch_size):
    job_env = submitit.JobEnvironment()
    model = AutoModel.from_pretrained(model_path)
    model = model.cuda(job_env.global_rank)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    df_length = input_df.shape[0]
    if os.path.exists(output_file):
        ids_already_done, vectors_already_done = torch.load(output_file)
    else:
        ids_already_done = torch.IntTensor()
        vectors_already_done = torch.FloatTensor()
    df_iterator = lazy_groups_of(input_df.to_dict(orient='records'), batch_size)
    for batch_json in tqdm(df_iterator,
                            total=df_length // batch_size):
        lines = [x['text'] for x in batch_json if x['text'] and x['text'] != '\n']

        input_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        last_non_masked_idx = torch.sum(input_ids['attention_mask'], dim=1) - 1
        # last_non_masked_idx = last_non_masked_idx.view(-1, 1).repeat(1, 768).unsqueeze(1).cuda()
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            vectors_ = model(**input_ids)
            vs = []
            # for ix, idx in enumerate(last_non_masked_idx):
                # vs.append(out[0][ix, :idx, :])
            # for ix, idx in enumerate(last_non_masked_idx):
                # vs.append(torch.mean(out[0][ix, :idx, :], 0).unsqueeze(0)) # Models outputs are now tuples
            # vectors_ = out[0].gather(1, last_non_masked_idx).squeeze(1) # Models outputs are now tuples
            # vectors_ = torch.cat(vs, 0)
        vectors.append(torch.mean(vectors_[0], 1).cpu())
        indices = torch.IntTensor([x['id'] for x in batch_json]).unsqueeze(-1)
        ids.append(indices)
    ids_ = torch.cat(ids, 0).cpu()
    vectors_ = torch.cat(vectors, 0).cpu()
    torch.save((torch.cat([ids_already_done, ids_], 0),
                torch.cat([vectors_already_done, vectors_], 0)), output_file)
    torch.save((torch.cat(ids, 0).cpu(), torch.cat(vectors, 0).cpu()), output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="path to huggingface model name (e.g. roberta-base) ")
    parser.add_argument("--output_dir", type=Path, required=True, help='path to output')
    parser.add_argument("--input_file", type=str, required=True, help='path to output')
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--num_shards', type=int, required=False, default=10)
    parser.add_argument('--log_dir', type=str, required=False, default="log_test")
    

    args = parser.parse_args()
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    vectors = []
    ids = []
    
    print(f'reading data from {args.input_file}...')
    df = pd.read_json(args.input_file, lines=True)
    df_chunks = np.array_split(df, args.num_shards)
    output_files = [args.output_dir  / f"{ix}.pt" for ix in range(len(df_chunks))]    

    log_folder = f"{args.log_dir}/%j"
    
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(slurm_array_parallelism=args.num_shards,
                               timeout_min=60,
                               slurm_partition="dev",
                               gpus_per_node=1)
    jobs = executor.map_array(extract,
                              df_chunks,
                              output_files,
                              [args.model] * len(df_chunks),
                              [args.batch_size] * len(df_chunks))
    print('submitted job!')
    num_finished = sum(job.done() for job in jobs)
    pbar = tqdm(total=len(jobs))
    pbar.set_description('job completion status')

    while num_finished < len(jobs):
        # wait and check how many have finished
        time.sleep(5)
        curr_finished = sum(job.done() for job in jobs)
        if curr_finished != num_finished:
            num_finished = curr_finished
            pbar.update(1)
    pbar.close()