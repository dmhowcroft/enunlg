import collections
import logging
import os
import random

import omegaconf
import torch

RegexRule = collections.namedtuple('RegexRule', ("match_expression", "replacement_expression"))


def count_parameters(model, log_table=True, print_table=False):
    """
    Based on https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model,
    forwarded to me by Jonas Groschwitz
    """
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if log_table:
        logging.info(table)
        logging.info(f"Total Trainable Params: {total_params}")
    if print_table:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params


def log_list_of_tensors_sizes(list_of_tensors, level=logging.DEBUG) -> None:
    logging.log(level, f"{len(list_of_tensors)=}")
    for task in list_of_tensors:
        logging.log(level, f"{task.size()}")


def log_sequence(seq, indent="") -> None:
    for element in seq:
        logging.info(f"{indent}{element}")


def save_config(config: omegaconf.DictConfig, hydra_config: omegaconf.DictConfig) -> None:
    with open(os.path.join(hydra_config.runtime.output_dir, 'hydra_config.yaml'), 'w') as hydra_file:
        hydra_file.write(omegaconf.OmegaConf.to_yaml(hydra_config))
    with open(os.path.join(hydra_config.runtime.output_dir, 'run_config.yaml'), 'w') as config_file:
        config_file.write(omegaconf.OmegaConf.to_yaml(config))


def set_random_seeds(seed) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
