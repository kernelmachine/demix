import wandb
api = wandb.Api()
import pandas as pd

# Project is specified by <entity/project-name>
runs = api.runs("suching/gpt3_experiments")
summary_list = [] 
config_list = [] 
name_list = [] 
rows = []
model = 'data_parallel_cord19_full'
for run in runs: 
    if run.name == model:
        for idx, row in run.history(keys=["valid_cord19/ppl", "train_inner/wall"]).iterrows():
            rows.append(row)
df = pd.concat(rows, 1).T
df.to_json(f"{model}.jsonl", lines=True, orient='records')
