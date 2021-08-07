#!/usr/bin/env python

import argparse
import collections

import torch
import numpy as np
import scipy.stats

from examples.few_shot.models import load_lm_and_run_func
from examples.few_shot.tasks import get_task_class_by_name, get_all_tasks
from examples.few_shot.templates import get_template_class_by_name
from examples.few_shot.predictors import get_predictor_class_by_name

def print_r0(x, file=None):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(x, file=file)


def cli_main():
    """Example usage:
    python -m examples.few_shot.gpt3_eval --model-name 124M --tasks copa cb --nb-few-shot-samples-values 0 1 32
    python -m examples.few_shot.gpt3_eval --model-name 124M --tasks copa cb --nb-few-shot-samples-values 0 1 32
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tasks", default=None, nargs="+")
    parser.add_argument("--predictor-name", default="clmprompting")
    parser.add_argument("--nb-few-shot-samples-values", type=int, default=None, nargs="+",
                        help="subsample K examples from the training set for one-shot or "
                             "few-shot learning")
    parser.add_argument("--uniform-sampling", action="store_true", help="take the same number of candidates per class when sampling")
    parser.add_argument("--beam-size", type=int, default=1, metavar='N',help="beam size for generative tasks")
    parser.add_argument("--num-trials", type=int, default=50, metavar='N',
                        help="when --nb-few-shot-samples is provided, repeat experiments N "
                             "times and report all results (few-shot experiments can have a high variance).")
    parser.add_argument("--train-sep", default=" ")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cause-conj", default=" because ")
    parser.add_argument("--effect-conj", default=" so ")
    parser.add_argument("--capitalization", default="correct",
                        choices=["correct", "bug", "upper", "lower"])
    for task in get_all_tasks():
        parser.add_argument(f"--{task}-template", metavar='TEMPLATE', default=None)
    args = parser.parse_args()
    args.train_sep = args.train_sep.replace("\\n", "\n")  # Newlines are escaped by argparse
    run_evaluations_from_model_name(**vars(args))


def run_evaluations_from_model_name(model_name, **kwargs):
    """Example usage:
    run_evaluations_from_model_name(model_name="124M", tasks=["copa", "cb"], nb_few_shot_samples_values=[0, 1, 32])
    """
    print_r0(f"model_name={model_name}")
    results = load_lm_and_run_func(run_evaluations, model_name, **kwargs)
    return results


def run_evaluations(model, tasks=None, nb_few_shot_samples_values=None, num_trials=50, **kwargs):
    if tasks is None:
        tasks = get_all_tasks()
    if nb_few_shot_samples_values is None:
        nb_few_shot_samples_values = [0, 1, 32]
    results = []
    for task_name in tasks:
        print_r0(f"task={task_name}")
        template_name = kwargs.get(f"{task_name}_template")
        if template_name is None:
            template_name = get_task_class_by_name(task_name).get_default_template_class().get_template_name()
        print_r0(f"template={template_name}")
        for nb_few_shot_samples in nb_few_shot_samples_values:
            print_r0(f"nb_few_shot_samples={nb_few_shot_samples}")
            num_trials_i = 1 if nb_few_shot_samples == 0 else num_trials
            metric2scores = collections.defaultdict(list)
            for seed in range(num_trials_i):
                for metric, score in run_evaluation(model=model, task_name=task_name, template_name=template_name, nb_few_shot_samples=nb_few_shot_samples, seed=seed, **kwargs).items():
                    metric2scores[metric].append(score)
            result_row = {
                "task": task_name,
                "template": template_name,
                "nb_few_shot_samples": nb_few_shot_samples,
            }
            for metric, scores in metric2scores.items():
                result_row[metric] = {
                    "scores": scores,
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "mean_confidence_interval": get_mean_confidence_interval(scores),
                }
            print_r0(f"results={result_row}\n")
            results.append(result_row)
    return results


def run_evaluation_from_model_name(model_name, **kwargs):
    print_r0(f"model_name={model_name}")
    result = load_lm_and_run_func(run_evaluation, model_name, **kwargs)
    return result


def run_evaluation(model, task_name, template_name, predictor_name, nb_few_shot_samples, uniform_sampling, seed=0, **kwargs):
    """Run single evaluation"""
    task = get_task_class_by_name(task_name=task_name).from_kwargs(**kwargs)
    task = task.get_random_subset(train_size=nb_few_shot_samples, valid_size=nb_few_shot_samples, uniform_sampling=uniform_sampling, seed=seed)
    template = get_template_class_by_name(template_name=template_name).from_kwargs(**kwargs)
    predictor = get_predictor_class_by_name(predictor_name=predictor_name).from_kwargs(model=model, task=task, template=template, **kwargs)
    predictions = predictor.predict(task.eval_samples)
    return {metric.name: metric.score(task.eval_samples, predictions) for metric in task.metrics}


def get_mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    a = 1.0 * data
    a = a[~np.isnan(a)]
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


if __name__ == "__main__":
    cli_main()
