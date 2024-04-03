"""Script for running the TGen NLU classifier."""

from pathlib import Path

import logging

import hydra
import numpy as np
import omegaconf
import torch

from enunlg.data_management.loader import load_data_from_config
from enunlg.nlu import binary_mr_classifier
from enunlg.normalisation.tokenisation import TGenTokeniser

import enunlg
import enunlg.data_management.e2e_challenge as e2e
import enunlg.embeddings.binary
import enunlg.meaning_representation.dialogue_acts as das
import enunlg.trainer.binary_mr_classifier
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger("enunlg-scripts.tgen_classifier")

MAX_INPUT_LENGTH_IN_KV_PAIRS = 10
MAX_INPUT_LENGTH_IN_INDICES = 3 * MAX_INPUT_LENGTH_IN_KV_PAIRS
HIDDEN_LAYER_SIZE = 50

EPOCHS = 20


def prep_tgen_text_integer_reps(input_corpus):
    """
    Expects a corpus in E2E challenge format. Returns the vocabulary the texts.
    (i.e. we create an integer-to-token reversible mapping for both separately)
    """
    return enunlg.vocabulary.TokenVocabulary([text.strip().split() for _, text in input_corpus])

SUPPORTED_DATASETS = {"e2e", "e2e-cleaned"}


def preprocess(corpus, preprocessing_config):
    if preprocessing_config.text.normalise == 'tgen':
        corpus = e2e.E2ECorpus([e2e.E2EPair(pair.mr, TGenTokeniser.tokenize(pair.text)) for pair in corpus])
    if preprocessing_config.text.delexicalise:
        logger.info('Applying delexicalisation...')
        if preprocessing_config.text.delexicalise.mode == 'split_on_caps':
            logger.info('...splitting on capitals in values')
            logger.info(f"...delexicalising: {preprocessing_config.text.delexicalise.slots}")
            corpus = e2e.E2ECorpus([e2e.delexicalise_exact_matches(pair,
                                                                   fields_to_delex=preprocessing_config.text.delexicalise.slots)
                                    for pair in corpus])
        else:
            message = "We can only handle the mode where we also check splitting on caps for values right now."
            raise ValueError(message)
    if preprocessing_config.mr.ignore_order:
        logger.info("Sorting slot-value pairs in the MR to ignore order...")
        corpus.sort_mr_elements()
    return corpus
    

@hydra.main(version_base=None, config_path='../config', config_name='tgen_classifier')
def tgen_classifier_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    # Pass the config to the appropriate function depending on what mode we are using
    if config.mode == "train":
        train_tgen_classifier(config)
    elif config.mode == "parameters":
        train_tgen_classifier(config, shortcircuit="parameters")
    elif config.mode == "test":
        test_tgen_classifier(config)
    else:
        message = "Expected config.mode to specify `train` or `parameters` modes."
        raise ValueError(message)


def train_tgen_classifier(config: omegaconf.DictConfig, shortcircuit=None):
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data, config.train.train_splits)
    corpus.print_summary_stats()
    print("____________")
    validation_corpus = load_data_from_config(config.data, config.train.dev_splits)
    validation_corpus.print_summary_stats()
    print("____________")

    corpus = preprocess(corpus, config.preprocessing)
    validation_corpus = preprocess(validation_corpus, config.preprocessing)

    logger.info("Preparing training data for PyTorch...")
    # Prepare mr/input integer representation
    token_int_mapper = prep_tgen_text_integer_reps(corpus)
    # Prepare bitvector encoding
    multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in corpus]
    bitvector_encoder = enunlg.embeddings.binary.DialogueActEmbeddings(multi_da_mrs, collapse_values=False)
    train_mr_bitvectors = [bitvector_encoder.embed_da(mr) for mr in multi_da_mrs]
    # Prepare text/output integer representation
    train_tokens = [text.strip().split() for _, text in corpus]
    text_lengths = [len(text) for text in train_tokens]
    train_text_ints = [token_int_mapper.get_ints_with_left_padding(text.split()) for _, text in corpus]
    logger.info(f"Text lengths: {min(text_lengths)} min, {max(text_lengths)} max, {sum(text_lengths)/len(text_lengths)} avg")
    logger.info("MRs as bitvectors:")
    enunlg.util.log_sequence(train_mr_bitvectors[:10], indent="... ")
    logger.info("and converting back from bitvectors:")
    enunlg.util.log_sequence([bitvector_encoder.embedding_to_string(bitvector) for bitvector in train_mr_bitvectors[:10]], indent="... ")
    logger.info(f"Text vocabulary has {token_int_mapper.max_index + 1} unique tokens")
    logger.info("The reference texts for these MRs:")
    enunlg.util.log_sequence(train_tokens[:10], indent="... ")
    logger.info("The same texts as lists of vocab indices")
    enunlg.util.log_sequence(train_text_ints[:10], indent="... ")

    logger.info("Preparing neural network...")
    tgen_classifier = binary_mr_classifier.TGenSemClassifier(token_int_mapper.size, bitvector_encoder.dimensionality, config.model)
    total_parameters = enunlg.util.count_parameters(tgen_classifier)
    if shortcircuit == 'parameters':
        exit()

    trainer = enunlg.trainer.binary_mr_classifier.BinaryMRClassifierTrainer(tgen_classifier, config.train, token_int_mapper, bitvector_encoder)

    training_pairs = [(torch.tensor(enc_emb, dtype=torch.long),
                       torch.tensor(dec_emb, dtype=torch.float))
                      for enc_emb, dec_emb in zip(train_text_ints, train_mr_bitvectors)]
    dev_text_ints = [token_int_mapper.get_ints_with_left_padding(text.split()) for _, text in validation_corpus]
    dev_multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in validation_corpus]
    dev_mr_bitvectors = [bitvector_encoder.embed_da(mr) for mr in dev_multi_da_mrs]
    validation_pairs = [(torch.tensor(enc_emb, dtype=torch.long),
                        torch.tensor(dec_emb, dtype=torch.float))
                        for enc_emb, dec_emb in zip(dev_text_ints, dev_mr_bitvectors)]

    logger.info(f"Running {config.train.num_epochs} epochs of {len(training_pairs)} iterations (with {len(validation_pairs)} validation pairs")
    losses_for_plotting = trainer.train_iterations(training_pairs, validation_pairs)

    tgen_classifier.save(Path(config.output_dir) / f'trained_{tgen_classifier.__class__.__name__}.nlg')
    token_int_mapper.save(Path(config.output_dir) / f'{token_int_mapper.__class__.__name__}.nlg')
    bitvector_encoder.save(Path(config.output_dir) / f'{bitvector_encoder.__class__.__name__}.nlg')


def test_tgen_classifier(config: omegaconf.DictConfig, shortcircuit=None):
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data, config.test.test_splits)
    corpus.print_summary_stats()
    print("____________")

    corpus = preprocess(corpus, config.preprocessing)

    # Load token_int_mapper
    token_int_mapper = enunlg.vocabulary.TokenVocabulary.load_from_dir(Path(config.test.classifier_file).parent / "TokenVocabulary.nlg")
    # Load bitvector_encoder
    bitvector_encoder = enunlg.embeddings.binary.DialogueActEmbeddings.load_from_dir(Path(config.test.classifier_file).parent / "DialogueActEmbeddings.nlg")

    # Prepare text/output integer representation
    test_tokens = [text.strip().split() for _, text in corpus]
    test_text_ints = [token_int_mapper.get_ints_with_left_padding(text) for text in test_tokens]
    multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in corpus]
    test_mr_bitvectors = [bitvector_encoder.embed_da(mr) for mr in multi_da_mrs]

    logger.info("Loading neural network...")
    tgen_classifier = binary_mr_classifier.TGenSemClassifier.load(config.test.classifier_file)
    total_parameters = enunlg.util.count_parameters(tgen_classifier)
    if shortcircuit == 'parameters':
        exit()

    test_pairs = [(torch.tensor(text_ints, dtype=torch.long),
                   torch.tensor(mr_bitvectors, dtype=torch.float))
                  for text_ints, mr_bitvectors in zip(test_text_ints, test_mr_bitvectors)]

    error = 0
    for i, o in test_pairs:
        prediction = tgen_classifier.predict(i).squeeze(0).squeeze(0).tolist()
        target_bitvector = np.round(o.tolist())
        output_bitvector = np.round(prediction)
        # print(prediction)
        # print(target_bitvector)
        # print(output_bitvector)
        current = enunlg.util.hamming_error(target_bitvector, output_bitvector)
        # print(current)
        error += current
    error = error / len(test_pairs)
    logger.info(f"Test error: {error:0.2f}")


if __name__ == "__main__":
    tgen_classifier_main()
