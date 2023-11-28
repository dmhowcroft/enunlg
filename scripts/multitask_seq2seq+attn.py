from typing import List

import logging
import os
import random

import matplotlib.pyplot as plt

import omegaconf
import hydra
import seaborn as sns
import torch

logging.basicConfig(encoding='utf-8', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")

from enunlg.data_management.pipelinecorpus import TextPipelineCorpus
from enunlg.trainer import MultiDecoderSeq2SeqAttnTrainer

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.encdec.multitask_seq2seq
import enunlg.vocabulary


class MultitaskSeq2SeqGenerator(object):
    def __init__(self, corpus: TextPipelineCorpus, model_config: omegaconf.DictConfig):
        """
        Create a multi-decoder seq2seq+attn model based on `corpus`.

        The first layer will be treated as input, subsequent layers will be treated as targets for decoding.
        At training time we use all the decoding layers, but at inference time we only decode at the final layer.

        :param corpus:
        """
        self.layers: List[str] = corpus.annotation_layers
        self.max_length_any_layer = corpus.max_layer_length
        logging.debug(f"{self.max_length_any_layer=}")
        self.vocabularies: Dict[str, enunlg.vocabulary.TokenVocabulary] = {layer: enunlg.vocabulary.TokenVocabulary(corpus.items_by_layer(layer)) for layer in self.layers} # type: ignore[misc]
        # There's definitely a cleaner way to do this, but we're lazy and hacky for a first prototype
        # We end up with a list of embeddings and a dict of list of embeddings to target
        self.input_embeddings = [torch.tensor(self.vocabularies[self.input_layer_name].get_ints_with_left_padding(item, self.max_length_any_layer), dtype=torch.long) for item in corpus.items_by_layer(self.input_layer_name)]
        self.output_embeddings = {layer: [torch.tensor(self.vocabularies[layer].get_ints(item), dtype=torch.long) for item in corpus.items_by_layer(layer)] for layer in self.decoder_target_layer_names}

        self.model = enunlg.encdec.multitask_seq2seq.MultiDecoderSeq2SeqAttn(self.layers, [self.vocabularies[layer].size for layer in self.vocabularies], model_config)

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


@hydra.main(version_base=None, config_path='../config', config_name='multitask_seq2seq+attn')
def multitask_seq2seq_attn_main(config: omegaconf.DictConfig):
    hydra_managed_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    seed = config.random_seed
    random.seed(seed)
    torch.manual_seed(seed)

    # TODO make more of the following dependent on config
    ee2e_corpus = ee2e.load_enriched_e2e(splits=("train",))
    for entry in ee2e_corpus[:6]:
        logging.info(entry)

    # Convert corpus to text pipeline corpus
    linearization_functions = {'raw_input': ee2e.linearize_slot_value_mr,
                               'selected_input': ee2e.linearize_slot_value_mr,
                               'ordered_input': ee2e.linearize_slot_value_mr,
                               'sentence_segmented_input': ee2e.linearize_slot_value_mr_seq,
                               'lexicalisation': lambda lex_string: lex_string.strip().split(),
                               'referring_expressions': lambda reg_string: reg_string.strip().split(),
                               'raw_output': lambda text: text.strip().split()}

    ee2e_corpus.print_summary_stats()
    print("____________")
    text_corpus = TextPipelineCorpus.from_existing(ee2e_corpus, mapping_functions=linearization_functions)
    text_corpus.print_summary_stats()
    text_corpus.print_sample(0, 100, 10)

    psg = MultitaskSeq2SeqGenerator(text_corpus, config.model)

    trainer = MultiDecoderSeq2SeqAttnTrainer(psg.model, config.mode.train, input_vocab=psg.vocabularies["raw_input"], output_vocab=psg.vocabularies["raw_output"])

    task_embeddings = []
    for idx in range(len(psg.input_embeddings)):
        task_embeddings.append([psg.output_embeddings[layer][idx] for layer in psg.decoder_target_layer_names])

    multitask_training_pairs = list(zip(psg.input_embeddings, task_embeddings))
    print(f"{multitask_training_pairs[0]=}")
    print(f"{len(multitask_training_pairs)=}")
    nine_to_one_split_idx = int(len(multitask_training_pairs) * 0.9)
    # losses_for_plotting = trainer.train_iterations(multitask_training_pairs[:90], multitask_training_pairs[90:100])
    losses_for_plotting = trainer.train_iterations(multitask_training_pairs[:nine_to_one_split_idx], multitask_training_pairs[nine_to_one_split_idx:])

    torch.save(psg.model.state_dict(), os.path.join(hydra_managed_output_dir, "trained-tgen-model.pt"))

    sns.lineplot(data=losses_for_plotting)
    plt.savefig(os.path.join(hydra_managed_output_dir, 'training-loss.png'))


if __name__ == "__main__":
    multitask_seq2seq_attn_main()
