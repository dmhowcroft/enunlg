from typing import Any, Dict, List, Optional

import logging
import os
import random

import omegaconf
import hydra
import torch

from enunlg.data_management.pipelinecorpus import TextPipelineCorpus
from enunlg.trainer.multitask_seq2seq import MultitaskTransformerTrainer

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.encdec.multitask_seq2seq
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger('enunlg-scripts.multitask_transformer')

SUPPORTED_DATASETS = {"enriched-e2e"}

# Convert corpus to text pipeline corpus
LINEARIZATION_FUNCTIONS = {'raw_input': ee2e.linearize_slot_value_mr,
                           'selected_input': ee2e.linearize_slot_value_mr,
                           'ordered_input': ee2e.linearize_slot_value_mr,
                           'sentence_segmented_input': ee2e.linearize_slot_value_mr_seq,
                           'lexicalisation': lambda lex_string: lex_string.strip().split(),
                           'referring_expressions': lambda reg_string: reg_string.strip().split(),
                           'raw_output': lambda text: text.strip().split()}


class MultitaskTransformer(torch.nn.Transformer):
    def __init__(self,
                 vocab_size,
                 # Args in the base class, with different defaults here
                 d_model: int = 64, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 256, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super(MultitaskTransformer, self).__init__(d_model, nhead,
                                                   num_encoder_layers, num_decoder_layers,
                                                   dim_feedforward,
                                                   dropout,
                                                   activation,
                                                   custom_encoder, custom_decoder,
                                                   layer_norm_eps,
                                                   batch_first, device, dtype)
        self.embeddings = torch.nn.Embedding(vocab_size, d_model)
        self.output_prediction = torch.nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, *args, **kwargs):
        # print(src.size())
        # print(tgt.size())
        src = self.embeddings(src).unsqueeze(0)
        tgt = self.embeddings(tgt).unsqueeze(0)
        # print(src.size())
        # print(tgt.size())
        xformer_output = super(MultitaskTransformer, self).forward(src, tgt, *args, **kwargs)
        return self.output_prediction(xformer_output)

    def train_step(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor, optimizer, criterion):
        optimizer.zero_grad()

        dec_outputs = self.forward(enc_emb, dec_emb)
        dec_targets = dec_emb.squeeze(0)
        # print(dec_outputs.squeeze(0).size())
        # print(dec_targets.size())
        loss = criterion(dec_outputs.squeeze(0), dec_targets)

        loss.backward()
        optimizer.step()
        # mean loss per word returned in order for losses for sents of diff lengths to be comparable
        return loss.item() / dec_targets.size(0)

    def generate(self, enc_emb, max_length=128):
        """Only implementing greedy for now."""
        with torch.no_grad():
            linear_preds = self.forward(enc_emb, enc_emb)
            # print(f"{linear_preds.size()=}")
            dec_outputs = torch.nn.functional.log_softmax(linear_preds.squeeze(0), dim=0)
            # print(f"{dec_outputs.size()=}")
            outputs = []
            for dec_output in dec_outputs:
                # print(f"{dec_output.size()=}")
                topv, topi = dec_output.data.topk(1)
                # print(topi)
                outputs.append(topi.item())
                if topi.item() == 2:
                    break
            return outputs


class MultitaskTransformerGenerator(object):
    def __init__(self, corpus: TextPipelineCorpus, model_config: omegaconf.DictConfig):
        """
        Create a multi-decoder seq2seq+attn model based on `corpus`.

        The first layer will be treated as input, subsequent layers will be treated as targets for decoding.
        At training time we use all the decoding layers, but at inference time we only decode at the final layer.

        :param corpus:
        """
        self.layers: List[str] = corpus.annotation_layers
        self.max_length_any_layer = corpus.max_layer_length
        logger.debug(f"{self.max_length_any_layer=}")
        self.vocabulary: Dict[str, enunlg.vocabulary.TokenVocabulary] = enunlg.vocabulary.TokenVocabulary(corpus.all_item_layer_iterator())
        # There's definitely a cleaner way to do this, but we're lazy and hacky for a first prototype
        # We end up with a list of embeddings and a dict of list of embeddings to target
        self.input_embeddings = [torch.tensor(self.vocabulary.get_ints_with_right_padding(item, 126), dtype=torch.long) for item in corpus.items_by_layer(self.input_layer_name)]
        self.output_embeddings = {layer: [torch.tensor(self.vocabulary.get_ints_with_right_padding(item, 126), dtype=torch.long) for item in corpus.items_by_layer(layer)] for layer in self.decoder_target_layer_names}

        self.model = MultitaskTransformer(self.vocabulary.size)

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


def set_random_seeds(seed) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_data_from_config(data_config) -> ee2e.EnrichedE2ECorpus:
    if data_config.corpus.name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {data_config.corpus.name}")
    if data_config.corpus.name == 'enriched-e2e':
        logger.info("Loading Enriched E2E Challenge Data...")
        return ee2e.load_enriched_e2e(data_config.corpus.splits)
    else:
        raise ValueError("We can only load the Enriched E2E dataset right now.")


def train_multitask_transformer(config: omegaconf.DictConfig, shortcircuit=None):
    hydra_managed_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    set_random_seeds(config.random_seed)

    ee2e_corpus = load_data_from_config(config.data)
    ee2e_corpus.print_summary_stats()
    print("____________")

    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    text_corpus = TextPipelineCorpus.from_existing(ee2e_corpus, mapping_functions=LINEARIZATION_FUNCTIONS)
    text_corpus.print_summary_stats()

    generator = MultitaskTransformerGenerator(text_corpus, config.model)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()


    task_embeddings = []
    for idx in range(len(generator.input_embeddings)):
        task_embeddings.append([generator.output_embeddings[layer][idx] for layer in generator.decoder_target_layer_names])

    e2e_training_pairs = list(zip(generator.input_embeddings, generator.output_embeddings['raw_output']))
    # e2e_training_pairs = e2e_training_pairs[:100]
    # multitask_training_pairs = list(zip(generator.input_embeddings, task_embeddings))


    trainer = MultitaskTransformerTrainer(generator.model, config.train, input_vocab=generator.vocabulary, output_vocab=generator.vocabulary)
    nine_to_one_split_idx = int(len(e2e_training_pairs) * 0.9)
    # losses_for_plotting = trainer.train_iterations(multitask_training_pairs[:90], multitask_training_pairs[90:100])
    losses_for_plotting = trainer.train_iterations(e2e_training_pairs[:nine_to_one_split_idx], e2e_training_pairs[nine_to_one_split_idx:])

    torch.save(generator.model.state_dict(), os.path.join(hydra_managed_output_dir, "trained-tgen-model.pt"))


@hydra.main(version_base=None, config_path='../config', config_name='multitask_seq2seq+attn')
def multitask_transformer_main(config: omegaconf.DictConfig):
    if config.mode == "train":
        train_multitask_transformer(config)
    elif config.mode == "parameters":
        train_multitask_transformer(config, shortcircuit="parameters")
    else:
        raise ValueError(f"Expected config.mode to specify `train` or `parameters` modes.")


if __name__ == "__main__":
    multitask_transformer_main()
