import logging
import os
import tarfile
import tempfile

from typing import Any, Dict, List

import omegaconf
import hydra
import torch

from sacrebleu import metrics as sm

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.data_management.enriched_webnlg
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.multitask_seq2seq
import enunlg.trainer.multitask_seq2seq
import enunlg.util
import enunlg.vocabulary
import enunlg.util

from enunlg.data_management.loader import load_data_from_config

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')


class MultitaskSeq2SeqGenerator(object):
    STATE_ATTRIBUTES = ('layers', 'vocabularies', 'max_length_any_layer', 'corpus_metadata', 'model')

    def __init__(self, corpus: enunlg.data_management.pipelinecorpus.TextPipelineCorpus, model_config: omegaconf.DictConfig):
        """
        Create a multi-decoder seq2seq+attn model based on `corpus`.

        The first layer will be treated as input, subsequent layers will be treated as targets for decoding.
        At training time we use all the decoding layers, but at inference time we only decode at the final layer.

        :param corpus:
        """
        self.layers: List[str] = corpus.annotation_layers
        self.vocabularies: Dict[str, enunlg.vocabulary.TokenVocabulary] = {layer: enunlg.vocabulary.TokenVocabulary(list(corpus.items_by_layer(layer))) for layer in self.layers} # type: ignore[misc]
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

    def _save_classname_to_dir(self, directory_path):
        with open(os.path.join(directory_path, "__class__.__name__"), 'w') as class_file:
            class_file.write(self.__class__.__name__)

    def save(self, filepath, tgz=True):
        os.mkdir(filepath)
        self._save_classname_to_dir(filepath)
        state = {}
        for attribute in self.STATE_ATTRIBUTES:
            curr_obj = getattr(self, attribute)
            save_method = getattr(curr_obj, 'save', None)
            print(curr_obj)
            print(save_method)
            if attribute == "vocabularies":
                # Special handling for the vocabularies
                state[attribute] = {}
                os.mkdir(f"{filepath}/{attribute}")
                for key in curr_obj:
                    state[attribute][key] = f"./{attribute}/{key}"
                    curr_obj[key].save(f"{filepath}/{attribute}/{key}", tgz=False)
            elif save_method is None:
                state[attribute] = curr_obj
            else:
                state[attribute] = f"./{attribute}"
                curr_obj.save(f"{filepath}/{attribute}", tgz=False)
        with open(os.path.join(filepath, "_save_state.yaml"), 'w') as state_file:
            state = omegaconf.OmegaConf.create(state)
            omegaconf.OmegaConf.save(state, state_file)
        if tgz:
            with tarfile.open(f"{filepath}.tgz", mode="x:gz") as out_file:
                out_file.add(filepath, arcname=os.path.basename(filepath))

    @classmethod
    def load(cls, filepath):
        if tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, 'r') as generator_file:
                tmp_dir = tempfile.mkdtemp()
                tarfile_member_names = generator_file.getmembers()
                generator_file.extractall(tmp_dir)
                root_name = tarfile_member_names[0].name[:-18]
                with open(os.path.join(tmp_dir, root_name, "__class__.__name__"), 'r') as class_name_file:
                    class_name = class_name_file.read().strip()
                    assert class_name == cls.__name__
                model = enunlg.encdec.multitask_seq2seq.DeepEncoderMultiDecoderSeq2SeqAttn.load_from_dir(os.path.join(tmp_dir, root_name, 'model'))
                dummy_pipeline_item = enunlg.data_management.pipelinecorpus.PipelineItem({layer_name: "" for layer_name in model.layer_names})
                dummy_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus([dummy_pipeline_item])
                dummy_corpus.pop()
                new_generator = cls(dummy_corpus, model.config)
                new_generator.model = model
                state_dict = omegaconf.OmegaConf.load(os.path.join(tmp_dir, root_name, "_save_state.yaml"))
                vocabs = {}
                for vocab in state_dict.vocabularies:
                    vocabs[vocab] = enunlg.vocabulary.TokenVocabulary.load_from_dir(os.path.join(tmp_dir, root_name, 'vocabularies', vocab))
                new_generator.vocabularies = vocabs
                return new_generator


def prep_embeddings(corpus, vocabularies, uniform_max_length=True):
    layer_names = list(vocabularies.keys())
    input_layer_name = layer_names[0]
    if uniform_max_length:
        max_length_any_layer = corpus.max_layer_length
    else:
        # e2e rdf test
        max_length_any_layer = 99
        # e2e normal test
        # max_length_any_layer = 78
    input_embeddings = [torch.tensor(vocabularies[input_layer_name].get_ints_with_left_padding(item, max_length_any_layer),
                                     dtype=torch.long) for item in corpus.items_by_layer(input_layer_name)]
    output_embeddings = {
        layer_name: [torch.tensor(vocabularies[layer_name].get_ints(item), dtype=torch.long) for item in
                corpus.items_by_layer(layer_name)] for layer_name in layer_names[1:]}
    return input_embeddings, output_embeddings


def validate_enriched_e2e(corpus) -> None:
    entries_to_drop = []
    for idx, entry in enumerate(corpus):
        # Some of the EnrichedE2E entries have incorrect semantics.
        # Checking for the restaurant name in the input selections is the fastest way to check.
        if 'name' in entry.raw_input and 'name' in entry.selected_input and 'name' in entry.ordered_input:
            pass
        else:
            entries_to_drop.append(idx)
    for idx in reversed(entries_to_drop):
        corpus.pop(idx)


def translate_e2e_to_rdf(corpus) -> None:
    for entry in corpus:
        agent = entry.raw_input['name']
        entry.raw_input = enunlg.util.mr_to_rdf(entry.raw_input)
        entry.selected_input = enunlg.util.mr_to_rdf(entry.selected_input)
        entry.ordered_input = enunlg.util.mr_to_rdf(entry.ordered_input)
        sentence_mrs = []
        for sent_mr in entry.sentence_segmented_input:
            sent_mr_dict = dict(sent_mr)
            sent_mr_dict['name'] = agent
            sentence_mrs.append(enunlg.util.mr_to_rdf(sent_mr_dict))
        entry.sentence_segmented_input = sentence_mrs


def train_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data)
    corpus.print_summary_stats()
    print("____________")

    # Drop entries that are missing data
    validate_enriched_e2e(corpus)

    if config.data.corpus.name == "enriched-e2e" and config.data.input_mode == "rdf":
        translate_e2e_to_rdf(corpus)

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

    input_embeddings, output_embeddings = prep_embeddings(text_corpus, generator.vocabularies)
    task_embeddings = []
    for idx in range(len(input_embeddings)):
        task_embeddings.append([output_embeddings[layer][idx] for layer in generator.layers[1:]])

    multitask_training_pairs = list(zip(input_embeddings, task_embeddings))
    # multitask_training_pairs = multitask_training_pairs[:100]
    print(f"{multitask_training_pairs[0]=}")
    print(f"{len(multitask_training_pairs)=}")
    nine_to_one_split_idx = int(len(multitask_training_pairs) * 0.9)
    trainer.train_iterations(multitask_training_pairs[:nine_to_one_split_idx], multitask_training_pairs[nine_to_one_split_idx:])

    generator.save(os.path.join(config.output_dir, f"trained_{generator.__class__.__name__}.nlg"))


def test_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data)
    corpus.print_summary_stats()
    print("____________")

    # Drop entries that are missing data
    validate_enriched_e2e(corpus)

    if config.data.corpus.name == "enriched-e2e" and config.data.input_mode == "rdf":
        translate_e2e_to_rdf(corpus)

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

    # drop entries that are too long
    indices_to_drop = []
    for idx, entry in enumerate(text_corpus):
        if len(entry['raw_input']) > generator.max_length_any_layer:
            indices_to_drop.append(idx)
            break

    logger.info(f"Dropping {len(indices_to_drop)} entries for having too long an input rep.")
    for idx in reversed(indices_to_drop):
        text_corpus.pop(idx)

    test_input, test_output = prep_embeddings(text_corpus, generator.vocabularies, False)
    test_ref = test_output['raw_output']
    logger.info(f"Num. input embeddings: {len(test_input)}")
    logger.info(f"Num. input refs: {len(test_ref)}")
    outputs = [generator.model.generate(embedding) for embedding in test_input]

    best_outputs = [" ".join(generator.vocabularies['raw_output'].get_tokens([int(x) for x in output])) for output in outputs]
    ref_outputs = [" ".join(generator.vocabularies['raw_output'].get_tokens([int(x) for x in output])) for output in test_ref]
    for best, ref in zip(best_outputs[:10], ref_outputs[:10]):
        logger.info(best)
        logger.info(ref)

    # Calculate BLEU compared to targets
    bleu = sm.BLEU()
    # We only have one reference per output
    bleu_score = bleu.corpus_score(best_outputs, [ref_outputs])
    logger.info(f"Current score: {bleu_score}")


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
        raise ValueError(f"Expected config.mode to specify `train` or `parameters` modes.")


if __name__ == "__main__":
    multitask_seq2seq_attn_main()
