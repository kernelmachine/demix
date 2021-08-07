# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

from fairseq.dataclass.configs import FairseqConfig


def merge_expert_and_shared_state(expert_state, shared_state):
    state = {}
    for key in ['cfg', 'args', 'extra_state', 'optimizer_history']:
        state[key] = expert_state[key]
    state['model'] = {**expert_state['model'], **shared_state['model']}

    if 'last_optimizer_state' in expert_state:
        state['last_optimizer_state'] = {}
        # REMOVED loss_scale!!
        for key in ['param_groups']:
            state['last_optimizer_state'][key] = expert_state['last_optimizer_state'][key]
        state['last_optimizer_state']['state'] = {
            **expert_state['last_optimizer_state']['state'],
            **shared_state['last_optimizer_state']['state'],
        }
    return state


def split_shared_and_expert_states(
    cfg: FairseqConfig,
    model, 
    optimizer
):
    shared_model_state_dict = OrderedDict()
    expert_model_state_dict = OrderedDict()
    shared_optimizer_state_dict = {}
    expert_optimizer_state_dict = {}
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    if getattr(cfg.model, "moe_freq", 0) > 0 or getattr(cfg.model, "desynchronize", False):
        param_name_to_is_expert = {}
        for name, param in model.named_parameters():
            param_name_to_is_expert[name] = hasattr(param, 'expert')

        for name, value in model_state_dict.items():
            if name in param_name_to_is_expert and param_name_to_is_expert[name]: 
                expert_model_state_dict[name] = value
            else:
                shared_model_state_dict[name] = value
    
    if cfg.common.fp16:
        expert_optimizer_state_dict['loss_scale'] = optimizer_state_dict['loss_scale']
        shared_optimizer_state_dict['loss_scale'] = optimizer_state_dict['loss_scale']

    for key in ['param_groups']:
        expert_optimizer_state_dict[key] = optimizer_state_dict[key]
        shared_optimizer_state_dict[key] = optimizer_state_dict[key]
    
    param_mappings = {}
    param_id_to_is_expert = {}
    start_index = 0
    for group in optimizer.param_groups:
        # nonlocal start_index
        packed = {k: v for k, v in group.items() if k != 'params'}
        for i, p in enumerate(group['params'], start_index):
            if id(p) not in param_mappings:
                param_mappings.update({id(p): i})
                param_id_to_is_expert[i] = hasattr(p, 'expert')
        packed['params'] = [param_mappings[id(p)] for p in group['params']]
        start_index += len(packed['params'])
        # return packed
        
    # param_groups = [pack_group(g) ]
    expert_optimizer_state_dict['state'] = {
        k: v for k, v in optimizer_state_dict['state'].items()
        if param_id_to_is_expert[k]
    }
    shared_optimizer_state_dict['state'] = {
        k: v for k, v in optimizer_state_dict['state'].items()
        if not param_id_to_is_expert[k]
    }
    return (
        (shared_model_state_dict, shared_optimizer_state_dict),
        (expert_model_state_dict, expert_optimizer_state_dict),
    )
