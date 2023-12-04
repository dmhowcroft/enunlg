from typing import List, Tuple, Optional

import logging
import random
import time

from sacrebleu import metrics as sm

import omegaconf
import torch

from enunlg.trainer.base import BasicTrainer

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')


class MultiDecoderSeq2SeqAttnTrainer(BasicTrainer):
    def __init__(self,
                 model,
                 training_config=None,
                 input_vocab=None,
                 output_vocab=None):
        if training_config is None:
            # Set defaults
            training_config = omegaconf.DictConfig({"num_epochs": 20,
                                                    "record_interval": 0.1,
                                                    "shuffle": True,
                                                    "batch_size": 1,
                                                    "optimizer": "adam",
                                                    "learning_rate": 0.001,
                                                    "learning_rate_decay": 0.5  # TGen used 0.0
                                                   })
        super().__init__(model, training_config)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self._curr_epoch = -1
        self._early_stopping_scores = [float('-inf')] * 5
        self._early_stopping_scores_changed = -1

    def _log_examples_this_interval(self, pairs):
        if self.input_vocab is None and self.output_vocab is None:
            super()._log_examples_this_interval(pairs)
        else:
            for i, o in pairs:
                logger.info("An example!")
                logger.info(f"Input:  {self.input_vocab.pretty_string(i.tolist())}")
                logger.info(f"Ref:    {self.output_vocab.pretty_string(o[-1].tolist())}")
                logger.info(f"Output: {self.output_vocab.pretty_string(self.model.generate(i))}")

    def train_iterations(self,
                         pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                         validation_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> List[float]:
        """
        Run `epochs` training epochs over all training pairs, shuffling pairs in place each epoch.

        :param pairs: input and output indices for embeddings
        :param validation_pairs: input and output indices for embeddings to be used in the validation step
        :return: list of average loss for each `record_interval` for each epoch
        """
        start_time = time.time()
        prev_chunk_start_time = start_time
        loss_this_interval = 0
        loss_to_plot = []
        # self.tb_writer.add_graph(self.model, [x for x, _ in pairs[:10]])

        # Add handling for proportional recording intervals
        if 0 < self.record_interval < 1:
            record_interval = int(self.record_interval * len(pairs))
        else:
            record_interval = self.record_interval

        stages = ['final', 'initial', 'weighted', 'all_balanced']
        for epoch in range(self.epochs):
            logger.info(f"Beginning epoch {epoch}...")
            self._curr_epoch = epoch
            self._log_epoch_begin_stats()
            random.shuffle(pairs)
            stage = stages[epoch % 4]
            stage = 'all_balanced'
            for index, (enc_emb, dec_emb) in enumerate(pairs, start=1):
                loss = self.model.train_step(enc_emb, dec_emb, self.optimizer, self.loss)
                self.log_training_loss(float(loss), epoch * len(pairs) + index)
                self.log_parameter_gradients(epoch * len(pairs) + index)

                loss_this_interval += loss
                if index % record_interval == 0:
                    avg_loss = loss_this_interval / record_interval
                    loss_this_interval = 0
                    logger.info("------------------------------------")
                    logger.info(f"{index} iteration mean loss = {avg_loss}")
                    logger.info(f"Time this chunk: {time.time() - prev_chunk_start_time}")
                    prev_chunk_start_time = time.time()
                    loss_to_plot.append(avg_loss)
                    self._log_examples_this_interval(pairs[:10])
            if validation_pairs is not None:
                logger.info("Checking for early stopping!")
                # Add check for minimum number of passes
                if self.early_stopping_criterion_met(validation_pairs):
                    break
            self.scheduler.step()
            logger.info("============================================")
            self.tb_writer.flush()
        logger.info("----------")
        logger.info(f"Training took {(time.time() - start_time) / 60} minutes")
        self.tb_writer.close()
        return loss_to_plot

    def sample_generations_and_references(self, pairs) -> Tuple[List[str], List[str]]:
        best_outputs = []
        ref_outputs = []
        for in_indices, out_indices in pairs:
            curr_outputs = self.model.generate(in_indices)
            best_outputs.append(self.output_vocab.pretty_string(curr_outputs))
            ref_outputs.append(self.output_vocab.pretty_string(out_indices[-1].tolist()))
        return best_outputs, ref_outputs

    def early_stopping_criterion_met(self, validation_pairs):
        # TGen uses BLEU score for validation
        # Generate current realisations for MRs in validation pairs
        best_outputs, ref_outputs = self.sample_generations_and_references(validation_pairs)
        # Calculate BLEU compared to targets
        bleu = sm.BLEU()
        # We only have one reference per output
        bleu_score = bleu.corpus_score(best_outputs, [ref_outputs])
        logger.info(f"Current score: {bleu_score}")
        if bleu_score.score > self._early_stopping_scores[-1]:
            self._early_stopping_scores.append(bleu_score.score)
            self._early_stopping_scores = sorted(self._early_stopping_scores)[1:]
            self._early_stopping_scores_changed = self._curr_epoch
        # If BLEU score has changed recently, keep training
        # NOTE: right now we're using the length of _early_stopping_scores to effectively ensure a minimum of 5 epochs
        if self._curr_epoch - self._early_stopping_scores_changed < len(self._early_stopping_scores):
            return False
        # Otherwise, stop early
        else:
            logger.info("Scores have not improved recently on the validation set, so we are stopping training now.")
            return True
