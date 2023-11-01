from collections import defaultdict
from typing import Dict

import logging

import omegaconf
import torch

logging.basicConfig(encoding='utf-8', level=logging.INFO)

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.seq2seq as s2s
import enunlg.meaning_representation.slot_value
import enunlg.trainer
import enunlg.vocabulary


class TextPipelineCorpus(enunlg.data_management.pipelinecorpus.PipelineCorpus):
    pass

    @classmethod
    def from_existing(cls, corpus: enunlg.data_management.pipelinecorpus.PipelineCorpus, mapping_functions):
        out_corpus = TextPipelineCorpus(corpus)
        for item in out_corpus:
            for layer in item.layers:
                item[layer] = mapping_functions[layer](item[layer])
        return out_corpus


def max_length_any_input_layer(corpus):
    max_length = -1
    for item in corpus:
        for layer in item.layers:
            if len(item[layer]) > max_length:
                logging.debug(f"New longest field, this time a {layer}")
                logging.debug(item[layer])
                max_length = len(item[layer])
    return max_length


class PipelineSeq2SeqGenerator(object):
    def __init__(self, corpus: ee2e.EnrichedE2ECorpus):
        self.layers = corpus.layers
        self.pipeline = corpus.layer_pairs
        self.max_length_any_layer = max_length_any_input_layer(corpus)
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
                                                   "embedding_dim": 16,
                                                   "backprop": True},
                                    "cell": "lstm",
                                    "num_hidden_dims": 64},
                        "decoder": {"embeddings": {"type": "torch",
                                                   "embedding_dim": 16,
                                                   "backprop": True,
                                                   "padding_idx": 0,
                                                   "start_idx": 1,
                                                   "stop_idx": 2
                                                   },
                                    "cell": "lstm",
                                    "num_hidden_dims": 64}})
        training_config = omegaconf.DictConfig({"num_epochs": 5,
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
            trainer.train_iterations([(x[0], x[1]) for x in zip(self.input_embeddings[in_layer], self.output_embeddings[out_layer])])

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


def sanitize_slot_names(slot_name):
    return slot_name


def sanitize_values(value):
    return value.replace(" ", "_").replace("'", "_")


def linearize_slot_value_mr(mr: enunlg.meaning_representation.slot_value.SlotValueMR):
    tokens = ["<MR>"]
    for slot in mr:
        tokens.append(sanitize_slot_names(slot))
        tokens.append(sanitize_values(mr[slot]))
        tokens.append("<PAIR_SEP>")
    tokens.append("</MR>")
    return tokens


def linearize_slot_value_mr_seq(mrs):
    tokens = ["<SENTENCE>"]
    for mr in mrs:
        tokens.extend(linearize_slot_value_mr(mr))
        tokens.append("</SENTENCE>")
    return tokens


if __name__ == "__main__":
    corpus = ee2e.load_enriched_e2e(splits=("dev", ))
    for x in corpus[:6]:
        print(x)

    # Convert corpus to text pipeline corpus
    linearization_functions = {'raw_input': linearize_slot_value_mr,
                               'selected_input': linearize_slot_value_mr,
                               'ordered_input': linearize_slot_value_mr,
                               'sentence_segmented_input': linearize_slot_value_mr_seq,
                               'lexicalisation': lambda x: x.strip().split(),
                               'referring_expressions': lambda x: x.strip().split(),
                               'raw_output': lambda x: x.strip().split()}

    corpus.print_summary_stats()
    print("____________")
    text_corpus = TextPipelineCorpus.from_existing(corpus, mapping_functions=linearization_functions)
    text_corpus.print_summary_stats()

    psg = PipelineSeq2SeqGenerator(corpus)
    mr_input_vocab = psg.vocabularies["raw_input"]
    for entry in corpus:
        mr = entry.raw_input
        print(mr)
        print(psg.predict(torch.tensor(mr_input_vocab.get_ints_with_left_padding(mr, psg.max_length_any_layer), dtype=torch.long)))
        print("----")
