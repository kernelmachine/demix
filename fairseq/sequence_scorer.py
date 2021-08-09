# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
from fairseq import utils
from torch_ema import ExponentialMovingAverage
from fairseq.modules.sparsemax import Sparsemax

def print_r0(x, file=None):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(x, file=file)


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(
        self,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
    ):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.sparsemax = Sparsemax(dim=0)


    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]
        len_models = torch.distributed.get_world_size() if torch.distributed.is_initialized() else len(models)

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        model_probs = []
        for model in models:
            model.eval()
            if not kwargs.get('decoder_out'):
                decoder_out = model(**net_input)
            else:
                decoder_out = kwargs.get('decoder_out')
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=not kwargs.get('ensemble') and len_models == 1, sample=sample
                ).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if kwargs.get('ensemble'):
                model_probs.append(probs.unsqueeze(0))
            else:
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs.add_(probs)
                if attn is not None:
                    if torch.is_tensor(attn):
                        attn = attn.data
                    else:
                        attn = attn[0]
                    if avg_attn is None:
                        avg_attn = attn
                    else:
                        avg_attn.add_(attn)
        if len_models > 1 or kwargs.get('ensemble'):
            if kwargs.get('ensemble'):
                if kwargs.get('all_reduce'):
                    gather_probs = [torch.ones_like(model_probs[0]).cuda() for _ in range(len_models)]
                    torch.distributed.all_gather(gather_probs, model_probs[0])
                    model_probs = torch.cat(gather_probs, dim=0)
                else:
                    model_probs = torch.cat(model_probs, dim=0)
                ## simple averaging
                if kwargs.get('ensemble_average'):
                    avg_probs = torch.mean(model_probs, dim=0)
                    weights = torch.tensor([ 1 / len_models]).repeat(len_models, model_probs.shape[1], model_probs.shape[2]).to(model_probs)
                elif kwargs.get('ensemble_weighted_average'):
                    # get t-1 probabilities
                    weights = model_probs[:, :, :-1].clone()
                    # either supply a prior or use uniform prior
                    if kwargs.get('prior') is not None:
                        priors = kwargs['prior']
                        priors = priors.tolist()

                    else:
                        # uniform
                        priors = [1 / len_models] * len_models
                        temperature = 1

                    # calculate normalization
                    denom = weights.clone()
                    for ix, prior in enumerate(priors):
                        denom[ix, :].mul_(prior)

                    denom = denom.sum(0)

                    # calculate posterior
                    for ix, prior in enumerate(priors):
                        weights[ix, :].mul_(prior).div_(denom)

                    # add uniform posterior probability at start of block
                    if kwargs.get('prior') is not None:
                        beginning_weights = torch.tensor([ 1 / len_models]).repeat(len_models, model_probs.shape[1], 1).to(weights)
                    else:
                        beginning_weights = torch.tensor(priors).float().repeat(model_probs.shape[1], 1).t().unsqueeze(2).to(weights)

                    weights = torch.cat([beginning_weights, weights], -1)

                    ## get weighted mixture
                    avg_probs = torch.einsum("ebs,ebs->bs", (weights,model_probs))


                avg_probs.log_()
                if avg_attn is not None:
                    if kwargs['ensemble']:
                        avg_attn.div_(len_models)
                    else:
                        avg_attn.div_(len_models)

            else:
                weights = None
                avg_probs.div_(len_models)
                avg_probs.log_()
                if avg_attn is not None:
                    avg_attn.div_(len_models)
        else:
            weights = None
            avg_probs.div_(len_models)
            if avg_attn is not None:
                avg_attn.div_(len_models)
            src = avg_probs

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append(
                [
                    {
                        "tokens": ref,
                        "expert_probs": weights[:, :, -1] if weights is not None else None,
                        "score": score_i,
                        "attention": avg_attn_i,
                        "alignment": alignment,
                        "positional_scores": avg_probs_i,
                    }
                ]
            )
        return hypos
