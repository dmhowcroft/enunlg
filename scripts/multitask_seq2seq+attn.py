from copy import deepcopy
from pathlib import Path

import json
import logging
import os

import omegaconf
import hydra
import torch

from enunlg.convenience.binary_mr_classifier import FullBinaryMRClassifier
from enunlg.data_management.loader import load_data_from_config
from enunlg.generators.multitask_seq2seq import MultitaskSeq2SeqGenerator, SingleVocabMultitaskSeq2SeqGenerator

import enunlg.data_management.enriched_e2e
import enunlg.data_management.enriched_webnlg
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.multitask_seq2seq
import enunlg.trainer.multitask_seq2seq
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')


def train_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data, config.train.train_splits)
    corpus.print_summary_stats()
    print("____________")
    dev_corpus = load_data_from_config(config.data, config.train.dev_splits)
    dev_corpus.print_summary_stats()
    print("____________")

    if config.data.corpus.name == "e2e-enriched":
        # Drop entries that are missing data
        corpus.validate_enriched_e2e()
        dev_corpus.validate_enriched_e2e()

    slot_value_format_corpus = deepcopy(corpus)
    slot_value_format_dev_corpus = deepcopy(dev_corpus)
    if config.data.corpus.name == "e2e-enriched" and config.data.input_mode == "rdf":
        enunlg.util.translate_e2e_to_rdf(corpus)
        enunlg.util.translate_e2e_to_rdf(dev_corpus)
    elif config.data.corpus.name == "webnlg-enriched":
        sem_class_dict = json.load(Path("datasets/processed/enriched-webnlg.dbo-delex.70-percent-coverage.json").open('r'))
        sem_class_lower = {key.lower(): sem_class_dict[key] for key in sem_class_dict}
        corpus.delexicalise_with_sem_classes(sem_class_lower)

    if config.data.input_mode == "rdf":
        linearization_functions = enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS
        linearization_metadata = "enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS"
    elif config.data.input_mode == "e2e":
        linearization_functions = enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS
        linearization_metadata = "enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS"
    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    text_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus.from_existing(corpus, mapping_functions=linearization_functions)
    text_corpus.metadata['linearization_functions'] = linearization_metadata
    text_corpus.print_summary_stats()
    text_corpus.print_sample(0, 100, 10)

    dev_text_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus.from_existing(dev_corpus, mapping_functions=linearization_functions)
    dev_text_corpus.metadata['linearization_functions'] = linearization_metadata
    dev_text_corpus.print_summary_stats()
    dev_text_corpus.print_sample(0, 100, 10)
    # drop entries that are too long
    indices_to_drop = []
    for idx, entry in enumerate(dev_text_corpus):
        if len(entry['raw_input']) > config.model.max_input_length - 2:
            indices_to_drop.append(idx)
            break
    logger.info(f"Dropping {len(indices_to_drop)} entries from the validation set for having too long an input rep.")
    for idx in reversed(indices_to_drop):
        dev_text_corpus.pop(idx)

    # generator = SingleVocabMultitaskSeq2SeqGenerator(text_corpus, config.model)
    generator = MultitaskSeq2SeqGenerator(text_corpus, config.model)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    trainer = enunlg.trainer.multitask_seq2seq.MultiDecoderSeq2SeqAttnTrainer(generator.model, config.train,
                                                                              input_vocab=generator.vocabularies["raw_input"],
                                                                              output_vocab=generator.vocabularies["raw_output"])

    # Section to be commented out normally, but useful for testing on small datasets
    # tmp_train_size = 50
    # tmp_dev_size = 10
    # slot_value_format_corpus = slot_value_format_corpus[:tmp_train_size]
    # text_corpus = text_corpus[:tmp_train_size]
    # slot_value_format_dev_corpus = slot_value_format_dev_corpus[:tmp_dev_size]
    # dev_text_corpus = dev_text_corpus[:tmp_dev_size]

    input_embeddings, output_embeddings = generator.prep_embeddings(text_corpus, config.model.max_input_length - 2)
    task_embeddings = [[output_embeddings[layer][idx]
                        for layer in generator.layers[1:]]
                       for idx in range(len(input_embeddings))]
    multitask_training_pairs = list(zip(input_embeddings, task_embeddings))

    dev_input_embeddings, dev_output_embeddings = generator.prep_embeddings(dev_text_corpus, config.model.max_input_length - 2)
    dev_task_embeddings = [[dev_output_embeddings[layer][idx]
                            for layer in generator.layers[1:]]
                           for idx in range(len(dev_input_embeddings))]
    multitask_validation_pairs = list(zip(dev_input_embeddings, dev_task_embeddings))

    trainer.train_iterations(multitask_training_pairs, multitask_validation_pairs)

    ser_classifier = FullBinaryMRClassifier.load(config.test.classifier_file)
    logger.info("===============================================")
    logger.info("Calculating performance on the training data...")
    generator.evaluate(slot_value_format_corpus, text_corpus, ser_classifier)

    logger.info("===============================================")
    logger.info("Calculating performance on the validation data...")
    generator.evaluate(slot_value_format_dev_corpus, dev_text_corpus, ser_classifier)

    generator.save(os.path.join(config.output_dir, f"trained_{generator.__class__.__name__}.nlg"))


def test_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    rich_corpus = load_data_from_config(config.data, config.test.test_splits)
    rich_corpus.print_summary_stats()
    print("____________")


    if config.data.corpus.name == "e2e-enriched":
        # Drop entries that are missing data
        enunlg.data_management.enriched_e2e.validate_enriched_e2e(rich_corpus)

    if config.data.corpus.name == "e2e-enriched" and config.data.input_mode == "rdf":
        enunlg.util.translate_e2e_to_rdf(rich_corpus)
    elif config.data.corpus.name == "webnlg-enriched":
        sem_class_dict = json.load(Path("datasets/processed/enriched-webnlg.dbo-delex.70-percent-coverage.json").open('r'))
        sem_class_lower = {key.lower(): sem_class_dict[key] for key in sem_class_dict}
        rich_corpus.delexicalise_with_sem_classes(sem_class_lower)

    if config.data.input_mode == "rdf":
        linearization_functions = enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS
    elif config.data.input_mode == "e2e":
        linearization_functions = enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS
    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    text_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus.from_existing(rich_corpus, mapping_functions=linearization_functions)
    text_corpus.print_summary_stats()
    text_corpus.print_sample(0, 100, 10)

    generator = MultitaskSeq2SeqGenerator.load(config.test.generator_file)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    ser_classifier = FullBinaryMRClassifier.load(config.test.classifier_file)
    generator.evaluate(rich_corpus, text_corpus, ser_classifier)


@hydra.main(version_base=None, config_path='../config', config_name='multitask_seq2seq+attn')
def multitask_seq2seq_attn_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    # Pass the config to the appropriate function depending on what mode we are using
    if config.mode == "train":
        train_multitask_seq2seq_attn(config)
    elif config.mode == "parameters":
        train_multitask_seq2seq_attn(config, shortcircuit="parameters")
    elif config.mode == "test":
        test_multitask_seq2seq_attn(config)
    else:
        message = "Expected config.mode to specify `train` or `parameters` modes."
        raise ValueError(message)


if __name__ == "__main__":
    multitask_seq2seq_attn_main()
