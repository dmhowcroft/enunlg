"""Script for running the TGen NLU classifier."""

import logging

import hydra
import omegaconf
import torch

from enunlg.data_management.loader import load_data_from_config
from enunlg.nlu import binary_mr_classifier
from enunlg.normalisation.tokenisation import TGenTokeniser

import enunlg
import enunlg.data_management.e2e_challenge as e2e
import enunlg.embeddings.binary
import enunlg.meaning_representation.dialogue_acts as das
import enunlg.util

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
            logger.info(f'...splitting on capitals in values')
            logger.info(f"...delexicalising: {preprocessing_config.text.delexicalise.slots}")
            corpus = e2e.E2ECorpus([e2e.delexicalise_exact_matches(pair,
                                                                   fields_to_delex=preprocessing_config.text.delexicalise.slots)
                                    for pair in corpus])
        else:
            raise ValueError("We can only handle the mode where we also check splitting on caps for values right now.")
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
        # test_tgen_classifier(config)
        pass
    else:
        raise ValueError(f"Expected config.mode to specify `train` or `parameters` modes.")


def train_tgen_classifier(config: omegaconf.DictConfig, shortcircuit=None):
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data)
    corpus.print_summary_stats()
    print("____________")

    corpus = preprocess(corpus, config.preprocessing)

    logger.info("Preparing training data for PyTorch...")
    # Prepare mr/input integer representation
    token_int_mapper = prep_tgen_text_integer_reps(corpus)
    # Prepare onehot encoding
    multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in corpus]
    onehot_encoder = enunlg.embeddings.binary.DialogueActEmbeddings(multi_da_mrs, collapse_values=False)
    train_mr_onehots = [onehot_encoder.embed_da(mr) for mr in multi_da_mrs]
    # Prepare text/output integer representation
    train_tokens = [text.strip().split() for _, text in corpus]
    text_lengths = [len(text) for text in train_tokens]
    train_text_ints = [token_int_mapper.get_ints_with_left_padding(text.split()) for _, text in corpus]
    logger.info(f"Text lengths: {min(text_lengths)} min, {max(text_lengths)} max, {sum(text_lengths)/len(text_lengths)} avg")
    logger.info("MRs as one-hot vectors:")
    enunlg.util.log_sequence(train_mr_onehots[:10], indent="... ")
    logger.info("and converting back from one-hot vectors:")
    enunlg.util.log_sequence([onehot_encoder.embedding_to_string(onehot_vector) for onehot_vector in train_mr_onehots[:10]], indent="... ")
    logger.info(f"Text vocabulary has {token_int_mapper.max_index + 1} unique tokens")
    logger.info("The reference texts for these MRs:")
    enunlg.util.log_sequence(train_tokens[:10], indent="... ")
    logger.info("The same texts as lists of vocab indices")
    enunlg.util.log_sequence(train_text_ints[:10], indent="... ")


    logger.info(f"Preparing neural network...")
    tgen_classifier = binary_mr_classifier.TGenSemClassifier(token_int_mapper, onehot_encoder)

    training_pairs = [(torch.tensor(enc_emb, dtype=torch.long),
                       torch.tensor(dec_emb, dtype=torch.float))
                      for enc_emb, dec_emb in zip(train_text_ints, train_mr_onehots)]

    logger.info(f"Running {config.train.num_epochs} epochs of {len(training_pairs)} iterations (looking at each training pair once per epoch)")
    losses_for_plotting = tgen_classifier.train_iterations(training_pairs, config.train.num_epochs)

    torch.save(tgen_classifier.state_dict(), "trained-tgen_classifier-model.pt")


if __name__ == "__main__":
    tgen_classifier_main()
