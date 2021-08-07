# Pattern based finetuning for few-shot tasks

## Preprocess
```bash
# preprocess for a single pattern of a single task
./examples/few_shot/pet/scripts/preprocess_fewglue.sh <data_folder> <task_name> <pattern_id>

# preprocess for super glue finetuning baseline
./examples/few_shot/pet/scripts/preprocess_sg_baseline.sh <data_folder> <task_name> -1
```

## Training

## TODO list

### 1) adding patterns and tasks
### 2) ensemble across patterns
### 3) Multi-token verbalizes
### 4) Some refactor to clean up the code