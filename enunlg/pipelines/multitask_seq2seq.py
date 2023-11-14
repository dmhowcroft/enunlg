from typing import Dict, List

import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

import omegaconf
import torch

from enunlg.data_management.enriched_e2e import linearize_slot_value_mr, linearize_slot_value_mr_seq
from enunlg.data_management.pipelinecorpus import TextPipelineCorpus
from enunlg.trainer import MultiDecoderSeq2SeqAttnTrainer

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.multitask_seq2seq
import enunlg.meaning_representation.slot_value
import enunlg.trainer
import enunlg.vocabulary


class MultitaskSeq2SeqGenerator(object):
    def __init__(self, corpus: TextPipelineCorpus, model_config: omegaconf.DictConfig = None):
        """
        Create a multi-decoder seq2seq+attn model based on `corpus`.

        The first layer will be treated as input, subsequent layers will be treated as targets for decoding.
        At training time we use all the decoding layers, but at inference time we only decode at the final layer.

        :param corpus:
        """
        self.layers: List[str] = corpus.layers
        self.pipeline = corpus.layer_pairs
        self.max_length_any_layer = corpus.max_layer_length
        logging.debug(f"{self.max_length_any_layer=}")
        self.vocabularies: Dict[str, enunlg.vocabulary.TokenVocabulary] = {layer: enunlg.vocabulary.TokenVocabulary(corpus.items_by_layer(layer)) for layer in self.layers} # type: ignore[misc]
        # There's definitely a cleaner way to do this, but we're lazy and hacky for a first prototype
        # We end up with a list of embeddings and a dict of list of embeddings to target
        self.input_embeddings = [torch.tensor(self.vocabularies[self.input_layer_name].get_ints_with_left_padding(item, self.max_length_any_layer), dtype=torch.long) for item in corpus.items_by_layer(self.input_layer_name)]
        self.output_embeddings = {layer: [torch.tensor(self.vocabularies[layer].get_ints(item), dtype=torch.long) for item in corpus.items_by_layer(layer)] for layer in self.decoder_target_layer_names}

        # Initialize the NN model
        if model_config is None:
            model_config = omegaconf.DictConfig({"name": "seq2seq+attn",
                                                 "max_input_length": self.max_length_any_layer,
                                                 "encoder": {"embeddings": {"type": "torch",
                                                                            "embedding_dim": 50,
                                                                            "backprop": True},
                                                             "cell": "lstm",
                                                             "num_hidden_dims": 50},
                                                 })
            for layer_name in self.decoder_target_layer_names:
                model_config[f"decoder_{layer_name}"] = {"embeddings": {"type": "torch",
                                                                        "embedding_dim": 50,
                                                                        "backprop": True,
                                                                        "padding_idx": 0,
                                                                        "start_idx": 1,
                                                                        "stop_idx": 2},
                                                         "cell": "lstm",
                                                         "num_hidden_dims": 50}

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

    def initialize_seq2seq_modules(self, corpus):
        # Define a default model config; we'll improve this later
        # TODO make the model config different for each pair of IO-layers
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
                                                       model_config=multidecoder_model_config)
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


if __name__ == "__main__":
    ee2e_corpus = ee2e.load_enriched_e2e(splits=("dev",))
    for x in ee2e_corpus[:6]:
        print(x)

    # Convert corpus to text pipeline corpus
    linearization_functions = {'raw_input': linearize_slot_value_mr,
                               'selected_input': linearize_slot_value_mr,
                               'ordered_input': linearize_slot_value_mr,
                               'sentence_segmented_input': linearize_slot_value_mr_seq,
                               'lexicalisation': lambda lex_string: lex_string.strip().split(),
                               'referring_expressions': lambda reg_string: reg_string.strip().split(),
                               'raw_output': lambda text: text.strip().split()}

    ee2e_corpus.print_summary_stats()
    print("____________")
    text_corpus = TextPipelineCorpus.from_existing(ee2e_corpus, mapping_functions=linearization_functions)
    text_corpus.print_summary_stats()
    # text_corpus.print_sample(0, 100, 10)

    psg = MultitaskSeq2SeqGenerator(text_corpus)
    mr_input_vocab = psg.vocabularies["raw_input"]
    test_mr_embedding = psg.input_embeddings[0]
    test_task_embeddings = [psg.output_embeddings[layer][0] for layer in psg.decoder_target_layer_names]

    trainer = MultiDecoderSeq2SeqAttnTrainer(psg.model, None, input_vocab=psg.vocabularies["raw_input"], output_vocab=psg.vocabularies["raw_output"])

    nine_to_one_split_idx = int(len(text_corpus))
    task_embeddings = []
    for idx in range(len(psg.input_embeddings)):
        task_embeddings.append([psg.output_embeddings[layer][idx] for layer in psg.decoder_target_layer_names])
    multitask_training_pairs = list(zip(psg.input_embeddings, task_embeddings))
    print(multitask_training_pairs[0])
    print(len(multitask_training_pairs))
    trainer.train_iterations(multitask_training_pairs)  # [:nine_to_one_split_idx], multitask_training_pairs[nine_to_one_split_idx:])

    # for entry in text_corpus[:10]:
    #     mr = entry.raw_input
    #     print(mr)
    #     output_seq = psg.predict(torch.tensor(mr_input_vocab.get_ints_with_left_padding(mr, psg.max_length_any_layer), dtype=torch.long))
    #     print(psg.vocabularies['raw_output'].get_tokens(output_seq))
    #     print("----")
