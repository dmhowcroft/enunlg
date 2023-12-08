import logging
import os

from typing import List

import omegaconf
import hydra
import torch

from sacrebleu import metrics as sm

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.multitask_seq2seq
import enunlg.trainer.multitask_seq2seq
import enunlg.util
import enunlg.vocabulary
import enunlg.util

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')

SUPPORTED_DATASETS = {"enriched-e2e"}

# Convert corpus to text pipeline corpus
LINEARIZATION_FUNCTIONS = {'raw_input': ee2e.linearize_slot_value_mr,
                           'selected_input': ee2e.linearize_slot_value_mr,
                           'ordered_input': ee2e.linearize_slot_value_mr,
                           'sentence_segmented_input': ee2e.linearize_slot_value_mr_seq,
                           'lexicalisation': lambda lex_string: lex_string.strip().split(),
                           'referring_expressions': lambda reg_string: reg_string.strip().split(),
                           'raw_output': lambda text: text.strip().split()}


class MultitaskSeq2SeqGenerator(object):
    def __init__(self, corpus: enunlg.data_management.pipelinecorpus.TextPipelineCorpus, model_config: omegaconf.DictConfig):
        """
        Create a multi-decoder seq2seq+attn model based on `corpus`.

        The first layer will be treated as input, subsequent layers will be treated as targets for decoding.
        At training time we use all the decoding layers, but at inference time we only decode at the final layer.

        :param corpus:
        """
        self.layers: List[str] = corpus.annotation_layers
        self.vocabularies: Dict[str, enunlg.vocabulary.TokenVocabulary] = {layer: enunlg.vocabulary.TokenVocabulary(corpus.items_by_layer(layer)) for layer in self.layers} # type: ignore[misc]
        # Store some basic information about the corpus
        self.max_length_any_layer = corpus.max_layer_length
        self.corpus_metadata = corpus.metadata
        self.model = enunlg.encdec.multitask_seq2seq.DeepEncoderMultiDecoderSeq2SeqAttn(self.layers, [self.vocabularies[layer].size for layer in self.vocabularies], model_config)

    @property
    def input_layer_name(self) -> str:
        return self.layers[0]

    @property
    def output_layer_name(self):
        return self.layers[-1]

    @property
    def decoder_target_layer_names(self):
        return self.layers[1:]

    @property
    def model_config(self):
        return self.model.config

    def predict(self, mr):
        return self.model.generate(mr)


def prep_embeddings(corpus, vocabularies, uniform_max_length=True):
    layer_names = list(vocabularies.keys())
    input_layer_name = layer_names[0]
    if uniform_max_length:
        max_length_any_layer = corpus.max_layer_length
    input_embeddings = [torch.tensor(vocabularies[input_layer_name].get_ints_with_left_padding(item, max_length_any_layer),
                                     dtype=torch.long) for item in corpus.items_by_layer(input_layer_name)]
    output_embeddings = {
        layer_name: [torch.tensor(vocabularies[layer_name].get_ints(item), dtype=torch.long) for item in
                corpus.items_by_layer(layer_name)] for layer_name in layer_names[1:]}
    return input_embeddings, output_embeddings


def load_data_from_config(data_config) -> ee2e.EnrichedE2ECorpus:
    if data_config.corpus.name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {data_config.corpus.name}")
    if data_config.corpus.name == 'enriched-e2e':
        logger.info("Loading Enriched E2E Challenge Data...")
        return ee2e.load_enriched_e2e(data_config.corpus.splits)
    else:
        raise ValueError("We can only load the Enriched E2E dataset right now.")


def train_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    ee2e_corpus = load_data_from_config(config.data)
    ee2e_corpus.print_summary_stats()
    print("____________")

    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    text_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus.from_existing(ee2e_corpus, mapping_functions=LINEARIZATION_FUNCTIONS)
    text_corpus.print_summary_stats()
    text_corpus.print_sample(0, 100, 10)

    generator = MultitaskSeq2SeqGenerator(text_corpus, config.model)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    trainer = enunlg.trainer.multitask_seq2seq.MultiDecoderSeq2SeqAttnTrainer(generator.model, config.train,
                                                                              input_vocab=generator.vocabularies["raw_input"],
                                                                              output_vocab=generator.vocabularies["raw_output"])

    input_embeddings, output_embeddings = prep_embeddings(text_corpus, generator.vocabularies)
    task_embeddings = []
    for idx in range(len(input_embeddings)):
        task_embeddings.append([output_embeddings[layer][idx] for layer in generator.layers[1:]])

    multitask_training_pairs = list(zip(input_embeddings, task_embeddings))
    print(f"{multitask_training_pairs[0]=}")
    print(f"{len(multitask_training_pairs)=}")
    nine_to_one_split_idx = int(len(multitask_training_pairs) * 0.9)
    trainer.train_iterations(multitask_training_pairs[:nine_to_one_split_idx], multitask_training_pairs[nine_to_one_split_idx:])

    torch.save(generator, os.path.join(config.output_dir, f"trained_{generator.__class__.__name__}.pt"))


def test_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    test_corpus = load_data_from_config(config)
    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    text_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus.from_existing(test_corpus, mapping_functions=LINEARIZATION_FUNCTIONS)

    generator: MultitaskSeq2SeqGenerator = torch.load(config.test.generator_file)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    test_input, test_output = prep_embeddings(test_corpus, generator.vocabularies)
    test_ref = [item_layer_embeddings[-1] for item_layer_embeddings in test_output]
    outputs = [generator.model.generate(embedding) for embedding in test_input]

    best_outputs = [" ".join(generator.vocabularies['raw_output'].get_tokens([int(x) for x in output])) for output in outputs]
    ref_outputs = [" ".join(generator.vocabularies['raw_output'].get_tokens([int(x) for x in output])) for output in test_ref]

    # Calculate BLEU compared to targets
    bleu = sm.BLEU()
    # We only have one reference per output
    bleu_score = bleu.corpus_score(best_outputs, [ref_outputs])
    logger.info(f"Current score: {bleu_score}")


@hydra.main(version_base=None, config_path='../config', config_name='multitask_seq2seq+attn')
def multitask_seq2seq_attn_main(config: omegaconf.DictConfig):
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    if config.mode == "train":
        train_multitask_seq2seq_attn(config)
    elif config.mode == "parameters":
        train_multitask_seq2seq_attn(config, shortcircuit="parameters")
    elif config.mode == "test":
        test_multitask_seq2seq_attn(config)
    else:
        raise ValueError(f"Expected config.mode to specify `train` or `parameters` modes.")


if __name__ == "__main__":
    multitask_seq2seq_attn_main()
