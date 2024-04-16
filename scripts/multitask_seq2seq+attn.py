import logging
import os

from sacrebleu import metrics as sm

import omegaconf
import hydra
import torch

from enunlg.convenience.binary_mr_classifier import FullBinaryMRClassifier
from enunlg.data_management.loader import load_data_from_config
from enunlg.generators.multitask_seq2seq import MultitaskSeq2SeqGenerator

import enunlg.data_management.enriched_e2e
import enunlg.data_management.enriched_webnlg
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.multitask_seq2seq
import enunlg.meaning_representation.dialogue_acts as das
import enunlg.trainer.multitask_seq2seq
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')


def train_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data, config.train.train_splits)
    corpus.print_summary_stats()
    print("____________")
    validation_corpus = load_data_from_config(config.data, config.train.dev_splits)
    validation_corpus.print_summary_stats()
    print("____________")

    # Drop entries that are missing data
    enunlg.data_management.enriched_e2e.validate_enriched_e2e(corpus)

    if config.data.corpus.name == "e2e-enriched" and config.data.input_mode == "rdf":
        enunlg.util.translate_e2e_to_rdf(corpus)

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

    generator = MultitaskSeq2SeqGenerator(text_corpus, config.model)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    trainer = enunlg.trainer.multitask_seq2seq.MultiDecoderSeq2SeqAttnTrainer(generator.model, config.train,
                                                                              input_vocab=generator.vocabularies["raw_input"],
                                                                              output_vocab=generator.vocabularies["raw_output"])

    input_embeddings, output_embeddings = generator.prep_embeddings(text_corpus)
    task_embeddings = [[output_embeddings[layer][idx]
                        for layer in generator.layers[1:]]
                       for idx in range(len(input_embeddings))]

    multitask_training_pairs = list(zip(input_embeddings, task_embeddings))
    # multitask_training_pairs = multitask_training_pairs[:100]
    print(f"{multitask_training_pairs[0]=}")
    print(f"{len(multitask_training_pairs)=}")
    nine_to_one_split_idx = int(len(multitask_training_pairs) * 0.9)
    trainer.train_iterations(multitask_training_pairs[:nine_to_one_split_idx], multitask_training_pairs[nine_to_one_split_idx:])

    generator.save(os.path.join(config.output_dir, f"trained_{generator.__class__.__name__}.nlg"))


def test_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data, config.test.test_splits)
    corpus.print_summary_stats()
    print("____________")

    # Drop entries that are missing data
    enunlg.data_management.enriched_e2e.validate_enriched_e2e(corpus)
    multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr in corpus.items_by_layer('raw_input')]

    if config.data.corpus.name == "e2e-enriched" and config.data.input_mode == "rdf":
        enunlg.util.translate_e2e_to_rdf(corpus)

    if config.data.input_mode == "rdf":
        linearization_functions = enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS
    elif config.data.input_mode == "e2e":
        linearization_functions = enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS
    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    text_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus.from_existing(corpus, mapping_functions=linearization_functions)
    text_corpus.print_summary_stats()
    text_corpus.print_sample(0, 100, 10)

    generator = MultitaskSeq2SeqGenerator.load(config.test.generator_file)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    max_input_length = generator.model.max_input_length - 2

    # drop entries that are too long
    indices_to_drop = []
    for idx, entry in enumerate(text_corpus):
        if len(entry['raw_input']) > max_input_length:
            indices_to_drop.append(idx)
            break

    logger.info(f"Dropping {len(indices_to_drop)} entries for having too long an input rep.")
    for idx in reversed(indices_to_drop):
        text_corpus.pop(idx)

    test_input, test_output = generator.prep_embeddings(text_corpus, max_input_length)
    test_ref = test_output['raw_output']
    logger.info(f"Num. input embeddings: {len(test_input)}")
    logger.info(f"Num. input refs: {len(test_ref)}")
    outputs = [generator.model.generate(embedding) for embedding in test_input]

    best_outputs = [" ".join(generator.vocabularies['raw_output'].get_tokens([int(x) for x in output])) for output in outputs]
    ref_outputs = [" ".join(generator.vocabularies['raw_output'].get_tokens([int(x) for x in output[1:]])).replace(" @ ", " ") for output in test_ref]
    for best, ref in zip(best_outputs[:10], ref_outputs[:10]):
        logger.info(best)
        logger.info(ref)

    # Calculate BLEU compared to targets
    bleu = sm.BLEU()
    # We only have one reference per output
    bleu_score = bleu.corpus_score(best_outputs, [ref_outputs])
    logger.info(f"Current score: {bleu_score}")

    # Estimate SER using classifier
    classifier = FullBinaryMRClassifier.load(config.test.classifier_file)
    test_tokens = [text.strip().split() for text in best_outputs]
    test_text_ints = [classifier.text_vocab.get_ints(text) for text in test_tokens]
    test_mr_bitvectors = [classifier.binary_mr_vocab.embed_da(mr) for mr in multi_da_mrs]
    ser_pairs = [(torch.tensor(text_ints, dtype=torch.long),
                  torch.tensor(mr_bitvectors, dtype=torch.float))
                 for text_ints, mr_bitvectors in zip(test_text_ints, test_mr_bitvectors)]

    logger.info(f"Test error: {classifier.evaluate(ser_pairs):0.2f}")


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
