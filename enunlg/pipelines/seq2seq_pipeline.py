from collections import defaultdict
from typing import Dict

import logging

import omegaconf
import torch

logging.basicConfig(encoding='utf-8', level=logging.INFO)

from enunlg.data_management.enriched_e2e import linearize_slot_value_mr, linearize_slot_value_mr_seq
from enunlg.data_management.pipelinecorpus import TextPipelineCorpus

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.seq2seq as s2s
import enunlg.meaning_representation.slot_value
import enunlg.trainer
import enunlg.vocabulary




class PipelineSeq2SeqGenerator(object):
    def __init__(self, corpus: TextPipelineCorpus):
        self.layers = corpus.layers
        self.pipeline = corpus.layer_pairs
        self.max_length_any_layer = corpus.max_layer_length
        logging.debug(f"{self.max_length_any_layer=}")
        self.modules = {layer_pair: None for layer_pair in self.pipeline}
        self.vocabularies: Dict[str, enunlg.vocabulary.TokenVocabulary] = {layer: None for layer in self.layers} # type: ignore[misc]
        # There's definitely a cleaner way to do this, but we're lazy and hacky for a first prototype
        self.input_embeddings = {layer: None for layer in self.layers}
        self.output_embeddings = {layer: None for layer in self.layers}
        self.initialize_vocabularies(corpus)
        self.initialize_embeddings(corpus)
        self.initialize_seq2seq_modules(corpus)

    def initialize_vocabularies(self, corpus):
        for layer in self.layers:
            self.vocabularies[layer] = enunlg.vocabulary.TokenVocabulary(corpus.items_by_layer(layer))

    def initialize_embeddings(self, corpus):
        for layer in self.layers:
            self.input_embeddings[layer] = [torch.tensor(self.vocabularies[layer].get_ints_with_left_padding(item, self.max_length_any_layer), dtype=torch.long) for item in corpus.items_by_layer(layer)]
            self.output_embeddings[layer] = [torch.tensor(self.vocabularies[layer].get_ints(item), dtype=torch.long) for item in corpus.items_by_layer(layer)]

    def initialize_seq2seq_modules(self, corpus):
        # Define a default model config; we'll improve this later
        # TODO make the model config different for each pair of IO-layers
        model_config = omegaconf.DictConfig({"name": "seq2seq+attn",
                        "max_input_length": max_length_any_input_layer(corpus),
                        "encoder": {"embeddings": {"type": "torch",
                                                   "embedding_dim": 50,
                                                   "backprop": True},
                                    "cell": "lstm",
                                    "num_hidden_dims": 50},
                        "decoder": {"embeddings": {"type": "torch",
                                                   "embedding_dim": 50,
                                                   "backprop": True,
                                                   "padding_idx": 0,
                                                   "start_idx": 1,
                                                   "stop_idx": 2
                                                   },
                                    "cell": "lstm",
                                    "num_hidden_dims": 50}})
        training_config = omegaconf.DictConfig({"num_epochs": 20,
                                                "record_interval": 410,
                                                "shuffle": True,
                                                "batch_size": 1,
                                                "optimizer": "adam",
                                                "learning_rate": 0.001,
                                                "learning_rate_decay": 0.5
        })
        for layer_pair in self.modules:
            in_layer, out_layer = layer_pair
            self.modules[layer_pair] = s2s.Seq2SeqAttn(self.vocabularies[in_layer].size,
                                                       self.vocabularies[out_layer].size,
                                                       model_config=model_config)
            trainer = enunlg.trainer.Seq2SeqAttnTrainer(self.modules[layer_pair],
                                                        training_config=training_config,
                                                        input_vocab=self.vocabularies[in_layer],
                                                        output_vocab=self.vocabularies[out_layer])
            corpus_pairs = [(x[0], x[1]) for x in zip(self.input_embeddings[in_layer], self.output_embeddings[out_layer])]
            idx_for_90_percent_split = int(len(corpus_pairs) * 0.9)
            trainer.train_iterations(corpus_pairs[:idx_for_90_percent_split],
                                     validation_pairs=corpus_pairs[idx_for_90_percent_split:])

    def predict(self, mr):
        # TODO we will need to add padding to the output of each layer before it can be used as input for the next
        curr_input = mr
        for layer_pair in self.modules:
            logging.debug(layer_pair)
            logging.debug(curr_input)
            curr_output = self.modules[layer_pair].generate(curr_input, max_length=self.max_length_any_layer)
            logging.debug(curr_output)
            padded_output = [self.vocabularies[layer_pair[0]].padding_token_int] * (self.max_length_any_layer - len(curr_output) + 2) + curr_output
            curr_input = torch.tensor(padded_output)
        return curr_output


def predict(model, mr):
    # TODO we will need to add padding to the output of each layer before it can be used as input for the next
    curr_input = mr
    for layer_pair in model.modules:
        print(f"Mapping from {layer_pair[0]} to {layer_pair[1]}.")
        print(" ".join(model.vocabularies[layer_pair[0]].get_tokens(curr_input.tolist())))
        curr_output = model.modules[layer_pair].generate(curr_input, max_length=model.max_length_any_layer)
        print(" ".join(model.vocabularies[layer_pair[1]].get_tokens(curr_output)))
        padded_output = [model.vocabularies[layer_pair[0]].padding_token_int] * (model.max_length_any_layer - len(curr_output) + 2) + curr_output
        curr_input = torch.tensor(padded_output)
    return curr_output


if __name__ == "__main__":
    corpus = ee2e.load_enriched_e2e(splits=("dev", ))
    for x in corpus[:6]:
        print(x)

    # Convert corpus to text pipeline corpus
    linearization_functions = {'raw_input': linearize_slot_value_mr,
                               'selected_input': linearize_slot_value_mr,
                               'ordered_input': linearize_slot_value_mr,
                               'sentence_segmented_input': linearize_slot_value_mr_seq,
                               'lexicalisation': lambda lex_string: lex_string.strip().split(),
                               'referring_expressions': lambda reg_string: reg_string.strip().split(),
                               'raw_output': lambda text: text.strip().split()}

    corpus.print_summary_stats()
    print("____________")
    text_corpus = TextPipelineCorpus.from_existing(corpus, mapping_functions=linearization_functions)
    text_corpus.print_summary_stats()
    text_corpus.print_sample(0, 100, 10)

    # psg = PipelineSeq2SeqGenerator(corpus)
    # mr_input_vocab = psg.vocabularies["raw_input"]
    # for entry in corpus[:10]:
    #     mr = entry.raw_input
    #     print(mr)
    #     output_seq = psg.predict(torch.tensor(mr_input_vocab.get_ints_with_left_padding(mr, psg.max_length_any_layer), dtype=torch.long))
    #     print(psg.vocabularies['raw_output'].get_tokens(output_seq))
    #     print("----")
