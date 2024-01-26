"""Script for evaluating saved outputs"""

from typing import Tuple

import logging
from typing import List

import hydra
import omegaconf

from sacrebleu import metrics as sm

import enunlg.util

logger = logging.getLogger(__name__)


def read_ref_inputs(ref_inputs_path: str) -> List[str]:
    # TODO change with real reader
    return read_texts(ref_inputs_path)


def read_texts(texts_path: str) -> List[str]:
    with open(texts_path, 'r') as text_file:
        return [line.strip() for line in text_file]


def calculate_bleu(texts: List[str], references: List[str], config: omegaconf.DictConfig) -> sm.BLEUScore:
    """Use sacrebleu to get a BLEU score"""
    class_kwargs = config.class_kwargs
    bleu = sm.BLEU(**class_kwargs)
    return bleu.corpus_score(texts, [references], config.n_bootstrap)


def ser_dict_score(dict_a, dict_b) -> Tuple[float, int, int, int, int, int]:
    """
    Compute the score between two dictionaries of slot-value counts.

    :param dict_a: dict of counts for diff slot-value combinations in text-being-evaluated
    :param dict_b: dict of counts for diff slot-value combinations in gold-standard reference
    :return: score, insertions, deletions, wrong
    """
    target_slots = 0
    for slot in dict_b:
        for val in dict_b[slot]:
            target_slots += dict_b[slot][val]
    insertions = 0
    deletions = 0
    wrong = 0
    correct = 0
    a_slots = set(dict_a.keys())
    b_slots = set(dict_b.keys())
    common_slots = a_slots.intersection(b_slots)
    a_only_slots = set(a_slots) - common_slots
    b_only_slots = set(b_slots) - common_slots
    for slot in common_slots:
        a_vals = dict_a[slot]
        b_vals = dict_b[slot]
        common_vals = set(a_vals.keys()).intersection(set(b_vals.keys()))
        a_only_vals = set(a_vals.keys()) - common_vals
        b_only_vals = set(b_vals.keys()) - common_vals
        for val in common_vals:
            if a_vals[val] == b_vals[val]:
                correct += 1
            elif a_vals[val] > b_vals[val]:
                insertions += 1
            else:
                deletions += 1
        if len(a_only_vals) == 1 and len(b_only_vals) == 1:
            wrong += 1
        else:
            for val in a_only_vals:
                insertions += a_vals[val]
            for val in b_only_vals:
                deletions += b_vals[val]
    for slot in a_only_slots:
        for val in dict_a[slot]:
            insertions += dict_a[slot][val]
    for slot in b_only_slots:
        for val in dict_b[slot]:
            insertions += dict_b[slot][val]
    return insertions + deletions + wrong / target_slots if target_slots > 0 else 0.0, insertions, deletions, wrong, correct, target_slots


def calculate_ser(texts, inputs, config):
    mr_dicts = []
    mr_from_inputs = []
    for text, input_mr in zip(texts, inputs):
        print(text)
        recognised_mr = enunlg.util.regex_extract_e2e_mr(text)
        print(recognised_mr)
        mr_dicts.append(recognised_mr)
        print(input_mr)
        recognised_mr = enunlg.util.regex_extract_e2e_mr(input_mr)
        print(recognised_mr)
        print("----")
        mr_from_inputs.append(recognised_mr)
    scores = [ser_dict_score(mr_from_text, mr_from_input)[0] for mr_from_text, mr_from_input in zip(mr_dicts, mr_from_inputs)]
    return sum(scores)/len(scores)


@hydra.main(version_base=None, config_path='../config', config_name='evaluate')
def evaluate_main(config: omegaconf.DictConfig) -> None:
    """
    Evaluate saved outputs, possibly with respect to some references.

    :param config: config with entries for reference and generated elements, as well as config for the eval metrics to use
    """
    # Collect reference inputs
    ref_inputs = read_ref_inputs(config.reference.inputs)
    # Collect reference texts
    ref_texts = read_texts(config.reference.texts)
    # Collect generated texts
    gen_texts = read_texts(config.generated.texts)

    # Calculate SER
    ser = calculate_ser(gen_texts, ref_inputs, config.ser)
    logger.info(f"SER:\t{ser}")
    # Calculate BLEU
    bleu = calculate_bleu(gen_texts, ref_texts, config.bleu)
    logger.info(f'BLEU:\t{bleu}')
    # Calculate SemClassifier Accuracy
    # Calculate BERTScore

if __name__ == "__main__":
    evaluate_main()
