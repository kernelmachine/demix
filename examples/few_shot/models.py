#!/usr/bin/env python
import os
import tempfile
from pathlib import Path
import pickle

import torch
from fairseq import distributed_utils, options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel


UNIDIR_LM_ROBERTA_DATA = {
    "124M": {
        "model_path": "/private/home/myleott/models/public_models/LM/roberta_lm.me_fp16.bm_none.tps1024.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.005.wu3000.dr0.1.atdr0.1.wd0.01.ms8.uf4.mu100000.s1.ngpu16/model.pt",
    },
    "354M": {
        "model_path": "/private/home/myleott/models/public_models/LM/roberta_lm.me_fp16.bm_none.tps1024.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.003.wu3000.dr0.1.atdr0.1.wd0.01.ms2.uf4.mu100000.s1.ngpu64/model.pt",
    },
    "355M_gpt3_setting": {
        "model_path": "/private/home/myleott/models/xlmg/unidir_lms/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms1.uf4.mu572204.s1.ngpu64/checkpoints/checkpoint_3_570000.pt",
        "dict_path": "/private/home/myleott/models/xlmg/unidir_lms/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms1.uf4.mu572204.s1.ngpu64/dict.txt",
    },
    "1.3B_gpt3_setting": {
        "model_path": "/private/home/myleott/models/xlmg/unidir_lms/1.3B/few_shot.roberta+cc100.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256/checkpoints/checkpoint_3_286000.pt",
    },
    "1.5B": {
        "model_path": "/private/home/myleott/models/public_models/LM/roberta_lm.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big.share.adafactor.lr0.0015.wu3000.wd0.01.dr0.1.atdr0.1.ms1.uf2.mu100000.s1.ngpu256/model.pt",
    },
    "1.5B_more_data": {
        "model_path": "/private/home/myleott/models/public_models/LM/roberta_plus_more_data_lm.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_mp_friendly.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0015.wu3000.wd0.01.dr0.1.atdr0.1.ctb.ms16.uf1.mu100000.s1.ngpu256/model.pt",
    },
    "11B": {
        "model_path": "/checkpoint/myleott/s3/models/model_parallel/megatron_11b/model.pt",
        # NOTE: we don't need model parallel here, inference should work on a 32GB V100
        #"enabled": torch.cuda.device_count() == 8,
        #"model_parallel_args": [
        #    "--model-parallel-size", "8",
        #    "--distributed-world-size", "8",
        #],
    },
    "seq2seq": {
        "model_path": "/private/home/jingfeidu/models/LM/few_shot.roberta+cc100.os.bm_none.tps2048.bart_base.seq2seq_lm.share.adam.b2_0.98.eps1e-08.cl1.0.lr5e-05.wu715.dr0.0.atdr0.1.wd0.01.ms4.uf1.mu572204.s1.min_enc_pct0.8.max_enc_pct0.8.ngpu64/model.pt",
    },
    "seq2seq_half": {
        "model_path": "/private/home/jingfeidu/models/LM/few_shot.roberta+cc100.os.bm_none.tps2048.bart_base.seq2seq_lm.share.adam.b2_0.98.eps1e-08.cl1.0.lr5e-05.wu715.dr0.0.atdr0.1.wd0.01.ms4.uf1.mu572204.s1.min_enc_pct0.8.max_enc_pct0.8.ngpu64/model_half.pt",
    },
    "routing_transformer_2048": {
        "model_path": "/private/home/jingfeidu/models/LM/few_shot.roberta+cc100.rt.fp16.tps2048.routing_transformer.aux_cross_entropy.adam.cl0.25.cosine.lr0.00025.s2.ngpu64/model.pt",
        "extra_args": [
            "--user-dir", "examples/routing_transformer_lm",
        ]
    },
    "routing_transformer_8192": {
        "model_path": "/private/home/jingfeidu/models/LM/few_shot.roberta+cc100.rt.fp16.routing_transformer.aux_cross_entropy.adam.cl0.25.cosine.lr0.00025.s2.ngpu64/model.pt",
        "extra_args": [
            "--user-dir", "examples/routing_transformer_lm",
        ]
    },
    "gpt3_355m_nodrop": {
        "path": "/private/home/myleott/models/xlmg/unidir_lms/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms1.uf4.mu572204.s1.ngpu64/checkpoint",
    },
    "gpt3_355m_drop": {
        "path": "/private/home/myleott/models/xlmg/unidir_lms/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu715.dr0.1.atdr0.1.wd0.01.ms1.uf4.mu572204.s1.ngpu64/checkpoint_best.pt",
    },
    "multilingual_seq2seq": {
        "model_path": "/checkpoint/artetxe/xlmg/run4/xlmg.cc5.os.langs_en_XX_fr_XX_es_XX_it_IT_de_DE.alpha_0.3.bm_none.tps1024.bart_large.seq2seq_lm.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms2.uf8.mu572204.s1.ngpu32/checkpoint_best.pt",
        "dict_path": "/datasets01/cc100-bin/072820/250/shard0/dict.txt",
        "model_overrides": {
            "bpe": "sentencepiece",
            "sentencepiece_model": "/private/home/louismartin/ts/resources/models/mbart/sentence.bpe.model",
        }
    },
    "moe_128exp_newdata": {
        "model_path": "/large_experiments/moe/namangoyal/checkpoints/moe_lms/the_pile/the_pile.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.001.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu35000.s1.ngpu128/checkpoint_7_35000.pt",
        "extra_args": [
            "--batch-size", "2", "--distributed-world-size", "128",  "--distributed-port", "15187", "--is-moe",
        ]
    },
    "moe_2exp": {
        "model_path": "/large_experiments/moe/shru/moe_lm/top2_2e_sv/top2_2e_sv.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu72000.s1.ngpu2/checkpoint_last.pt",
        "extra_args": [
            "--batch-size", "2", "--distributed-world-size", "2",  "--distributed-port", "15187", "--is-moe",
        ]
    },
}


def print_r0(x, file=None):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(x, file=file)


def pickle_dump(obj, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def load_lm_and_run_func(func, model_name, **kwargs):
    if model_name in UNIDIR_LM_ROBERTA_DATA:
        model_config = UNIDIR_LM_ROBERTA_DATA[model_name]
    elif os.path.exists(model_name):
        model_config = {"path": model_name}
    else:
        raise ValueError(f"unknown --model-name={model_name}")
    if not model_config.get("enabled", True):
        print_r0("skipping disabled model: " + model_name)
        return
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(
        parser,
        input_args=[
            os.path.dirname(model_config["model_path"]),  # data argument
        ] + model_config.get("model_parallel_args", []) + model_config.get("extra_args", []),
    )
    cfg = convert_namespace_to_omegaconf(args)
    return_value_path = Path(tempfile.mkstemp()[1])
    distributed_utils.call_main(
        main=_load_lm_and_run_func,
        cfg=cfg,
        config=model_config,
        return_value_path=return_value_path,
        func=func,
        **kwargs,
    )
    if cfg.distributed_training.device_id == 0:
        return_value = pickle_load(return_value_path)
        return_value_path.unlink()
    else:
        return_value = None
    return return_value


def _load_lm_and_run_func(fairseq_cfg, config, return_value_path, func, **kwargs):
    if torch.distributed.is_initialized():
        fairseq_cfg.checkpoint.checkpoint_suffix = f"-rank-{torch.distributed.get_rank()}"
        print_r0(f"suffix={fairseq_cfg.checkpoint.checkpoint_suffix}")
    # TODO: stop hard-coding moe_freq
    model = get_model(model_path=config["model_path"], dict_path=config.get("dict_path"), suffix=fairseq_cfg.checkpoint.checkpoint_suffix or "", moe_freq=2, **config.get("model_overrides", {}))
    model.half()
    model.cuda()
    model.eval()  # disable dropout
    model.cfg.dataset.max_tokens = get_or_infer_max_tokens(model, **kwargs)
    return_value = func(model=model, **kwargs)
    rank = fairseq_cfg.distributed_training.device_id
    if rank == 0:
        pickle_dump(return_value, return_value_path)


def get_model(model_path, dict_path=None, suffix="", bpe="gpt2", **kwargs):
    model_path = Path(model_path)
    # assert model_path.exists()
    model_name_or_path = str(model_path.parent)
    checkpoint_file = model_path.name
    data_name_or_path = "."
    if dict_path is not None:
        dict_path = Path(dict_path)
        assert dict_path.exists()
        # HACK: The setup_task method will look in the data_dir for dict.txt
        # https://github.com/pytorch/fairseq/blob/dea66cc294a18dd4d9e59aa0af8d51f951e83884/fairseq/tasks/language_modeling.py#L141
        data_name_or_path = str(dict_path.parent)
    model = BaseFairseqModel.from_pretrained(
        model_name_or_path=model_name_or_path,
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_name_or_path,
        suffix=suffix,
        # If the criterion in the checkpoint is set to
        # vocab_parallel_cross_entropy, then the output layer will skip the
        # gather_from_model_parallel_region call, which is necessary.
        # So here we override the criterion to force the output layer to call
        # gather_from_model_parallel_region.
        criterion="cross_entropy",
        bpe=bpe,
        **kwargs,
    )
    model.task = override_task_for_seq2seq_models(model)
    model.cfg.dataset.batch_size = None  # We will infer max_tokens at runtime
    return model


def override_task_for_seq2seq_models(model):
    # HACK: override task for some seq2seq models
    if model.cfg.model._name in ["bart_base", "bart_large"] and model.cfg.task._name in ["language_modeling", "multilingual_language_modeling"] and model.cfg.criterion._name == "seq2seq_lm":
        from fairseq import tasks
        task_config = model.cfg.task
        task_config._name = "seq2seq_lm"
        return tasks.setup_task(task_config, criterion_args=model.cfg.criterion)
    else:
        return model.task


def get_or_infer_max_tokens(model, **kwargs):
    if "max_tokens" in kwargs:
        model.cfg.dataset.max_tokens = kwargs["max_tokens"]
    if model.cfg.dataset.max_tokens is not None or model.cfg.dataset.batch_size is not None:
        return model.cfg.dataset.max_tokens
    return infer_max_tokens_before_oom(model)


def convert_max_positions_to_int(max_positions):
    if isinstance(max_positions, int):
        return max_positions
    elif isinstance(max_positions, tuple):  # For seq2seq models where it's a tuple for encoder and decoder
        # TODO: Ideally we could take the sum because tokens are spread accross encoder and decoder.
        # However it can impose a constraint on how the tokens are split w.r.t. encoder_resp.
        return min(max_positions)


def infer_max_tokens_before_oom(model):
    def is_max_tokens_oom(max_tokens):
        try:
            max_positions = convert_max_positions_to_int(model.max_positions)
            dummy_sample = model.decode([42] * (max_positions - 2))  # Minus 2 for special tokens in seq2seq
            input_texts = [dummy_sample for _ in range(int(candidate_max_tokens / max_positions))]
            model.score(input_texts)
            return False
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise e
            return True

    assert model.cfg.dataset.max_tokens is None
    assert model.cfg.dataset.batch_size is None
    print_r0("Infering max tokens for model...")
    candidate_max_tokens = convert_max_positions_to_int(model.max_positions)
    while not is_max_tokens_oom(candidate_max_tokens):
        candidate_max_tokens *= 2
    max_tokens = candidate_max_tokens // 2
    print_r0(f"Setting max_tokens to {max_tokens}")
    return max_tokens


def run_with_oom_catch(func, **kwargs):
    """Return results of func(**kwargs) or catches OOM exceptions and returns None.

    Useful to run the OOM catch in a compartimented function to release tensors when exiting the local scope.
    Catching OOM in the outer scope will indeed prevent the tensors that exist at the time of the error to be released.
    """
    try:
        return func(**kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e):
            raise e
        return None


def run_with_adaptative_max_tokens(model, func, **kwargs):
    """Runs func(model, **kwargs) or lower max_tokens and try again when OOM occurs. func should never return None."""
    results = run_with_oom_catch(func, **kwargs)
    if results is not None:
        return results
    else:
        print_r0(f"OOM: max_tokens={model.cfg.dataset.max_tokens} ==> max_tokens={model.cfg.dataset.max_tokens//2}")
        model.cfg.dataset.max_tokens //= 2
        return run_with_adaptative_max_tokens(model, func, **kwargs)
