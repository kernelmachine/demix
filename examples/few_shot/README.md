# XL Generative Zero-Shot Learner

## Example Usage
### CLI
```bash
$ python -m examples.few_shot.gpt3_eval --model-name 124M --tasks copa cb --nb-few-shot-samples-values 0 1 32 --num-trials 5 --train-sep "\n"

model_name=124M
Infering max tokens for model...
Setting max_tokens to 16384
task=copa
nb_few_shot_samples=0
100it [00:00, 838.46it/s]
results={'task': 'copa', 'nb_few_shot_samples': 0, 'scores': [64.0], 'mean': 64.0, 'std': 0.0, 'mean_confidence_interval': nan}

nb_few_shot_samples=1
100it [00:00, 680.98it/s]
results={'task': 'copa', 'nb_few_shot_samples': 1, 'scores': [62.0, 62.0, 65.0, 62.0, 60.0], 'mean': 62.2, 'std': 1.5999999999999999, 'mean_confidence_interval': 2.2211560841582387}

nb_few_shot_samples=32
OOM: max_tokens=16384 ==> max_tokens=8192
100it [00:02, 49.96it/s]
results={'task': 'copa', 'nb_few_shot_samples': 32, 'scores': [63.0, 65.0, 63.0, 64.0, 60.0], 'mean': 63.0, 'std': 1.6733200530681511, 'mean_confidence_interval': 2.3229406353851942}

task=cb
nb_few_shot_samples=0
56it [00:00, 190.88it/s]
results={'task': 'cb', 'nb_few_shot_samples': 0, 'scores': [42.857142857142854], 'mean': 42.857142857142854, 'std': 0.0, 'mean_confidence_interval': nan}

nb_few_shot_samples=1
56it [00:00, 133.04it/s]
results={'task': 'cb', 'nb_few_shot_samples': 1, 'scores': [41.07142857142857, 17.857142857142858, 41.07142857142857, 41.07142857142857, 12.5], 'mean': 30.71428571428571, 'std': 12.797480619406368, 'mean_confidence_interval': 17.76575121230725}

nb_few_shot_samples=32
56it [00:04, 12.43it/s]
results={'task': 'cb', 'nb_few_shot_samples': 32, 'scores': [42.857142857142854, 41.07142857142857, 41.07142857142857, 42.857142857142854, 41.07142857142857], 'mean': 41.78571428571428, 'std': 0.874817765279706, 'mean_confidence_interval': 1.2144417511754582}
```

### Python
```python
from examples.few_shot.gpt3_eval import run_evaluations_from_model_name

run_evaluations_from_model_name(model_name="124M", tasks=["copa", "cb"], nb_few_shot_samples_values=[0, 1, 32], num_trials=5, train_sep="\n")
```
