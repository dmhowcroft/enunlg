from typing import List, Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    import enunlg.encdec.sclstm

import logging
import random
import time

from torch.utils.tensorboard import SummaryWriter

import omegaconf
import sacrebleu.metrics as sm
import torch

import enunlg.encdec.seq2seq

logger = logging.getLogger(__name__)


class BasicTrainer(object):
    def __init__(self,
                 model,
                 training_config):
        """
        A basic class to be implemented with particular details specified for different NN models

        :param model: a PyTorch NN model to be trained
        :param training_config: details for how the model should be trained and what we should track
        """
        self.model = model
        self.config = training_config
        self.epochs = self.config.num_epochs
        self.record_interval = self.config.record_interval
        self.shuffle_order_each_epoch = self.config.shuffle

        # Initialize loss
        # TODO add support for different loss functions
        self.loss = torch.nn.CrossEntropyLoss()

        # Initialize optimizers
        self.learning_rate = self.config.learning_rate
        if self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported choice of optimizer. Expecting 'adam' or 'sgd' but got {self.config.optimizer}")

        # Initialize scheduler for optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=self.config.learning_rate_decay)
        self.tb_writer = SummaryWriter()

    def log_training_loss(self, loss, index):
        self.tb_writer.add_scalar(f'{self.__class__.__name__}-training_loss', loss, index)

    def log_parameter_gradients(self, index):
        for param, value in self.model.named_parameters():
            self.tb_writer.add_scalar(f"{self.__class__.__name__}-{param}-grad", torch.mean(value.grad), )

    def train_iterations(self, *args, **kwargs):
        # TODO increase consistency between SCLSTM and TGen training so we can pull things up to this level
        raise NotImplementedError("Use one of the subclasses, don't try to use this one directly")

    def _log_epoch_begin_stats(self):
        logger.info(f"Learning rate is now {self.learning_rate}")

    def _log_examples_this_interval(self, pairs):
        for i, o in pairs:
            logger.info("An example!")
            logger.info(f"Input:  {self.model.input_rep_to_string(i.tolist())}")
            logger.info(f"Ref:    {self.model.output_rep_to_string(o.tolist())}")
            logger.info(f"Output: {self.model.output_rep_to_string(self.model.generate(i))}")


class SCLSTMTrainer(BasicTrainer):
    def __init__(self,
                 model: "enunlg.encdec.sclstm.SCLSTMModel",
                 training_config=None):
        if training_config is None:
            # Set defaults
            training_config = omegaconf.DictConfig({"num_epochs": 20,
                                                    "record_interval": 519,
                                                    "shuffle": True,
                                                    "batch_size": 1,
                                                    "optimizer": "sgd",
                                                    "learning_rate": 0.1,
                                                    "learning_rate_decay": 0.5
                                                    })
        super().__init__(model, training_config)

        # Re-initialize loss using summation instead of mean
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def train_iterations(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                         teacher_forcing_rule=None) -> List[float]:
        """
        Run `epochs` training epochs over all training pairs, shuffling pairs in place each epoch.

        :param pairs: input and output indices for embeddings
        :param teacher_forcing_rule: what rule to use determining how much teacher forcing to use during training
        :return: list of average loss for each `record_interval` for each epoch
        """
        start_time = time.time()
        prev_chunk_start_time = start_time
        loss_this_interval = 0
        loss_to_plot = []

        prob_teacher_forcing = 1.0
        if teacher_forcing_rule is None:
            pass
        elif teacher_forcing_rule == 'reduce_over_scheduled_epochs':
            logger.info("Reducing teacher forcing linearly with number of (epochs / number of epochs scheduled)")
        else:
            logging.warning(f"Invalid value for teacher_forcing_rule: {teacher_forcing_rule}. Using default.")

        for epoch in range(self.epochs):
            if teacher_forcing_rule == 'reduce_over_scheduled_epochs':
                prob_teacher_forcing = 1.0 - epoch / self.epochs
            logger.info(f"Beginning epoch {epoch}...")
            self._log_epoch_begin_stats()
            random.shuffle(pairs)
            for index, (enc_emb, dec_emb) in enumerate(pairs, start=1):
                loss = self.model.train_step(enc_emb, dec_emb, self.optimizer, self.loss)
                self.log_training_loss(float(loss), epoch * len(pairs) + index)
                self.log_parameter_gradients(epoch * len(pairs) + index)
                loss_this_interval += loss
                if index % self.record_interval == 0:
                    avg_loss = loss_this_interval / self.record_interval
                    loss_this_interval = 0
                    logger.info("------------------------------------")
                    logger.info(f"{index} iteration mean loss = {avg_loss}")
                    logger.info(f"Time this chunk: {time.time() - prev_chunk_start_time}")
                    prev_chunk_start_time = time.time()
                    loss_to_plot.append(avg_loss)
                    self._log_examples_this_interval(pairs[:10])
            self.scheduler.step()
            logger.info("============================================")
        logger.info("----------")
        logger.info(f"Training took {(time.time() - start_time) / 60} minutes")
        self.tb_writer.close()
        return loss_to_plot


class Seq2SeqAttnTrainer(BasicTrainer):
    def __init__(self,
                 model: "enunlg.encdec.seq2seq.Seq2SeqAttn|enunlg.encdec.tgen.TGenEncDec",
                 training_config=None,
                 input_vocab=None,
                 output_vocab=None):
        if training_config is None:
            # Set defaults
            training_config = omegaconf.DictConfig({"num_epochs": 20,
                                                    "record_interval": 100,
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
                logger.info(f"Ref:    {self.output_vocab.pretty_string(o.tolist())}")
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

        for epoch in range(self.epochs):
            logger.info(f"Beginning epoch {epoch}...")
            self._curr_epoch = epoch
            self._log_epoch_begin_stats()
            random.shuffle(pairs)
            for index, (enc_emb, dec_emb) in enumerate(pairs, start=1):
                loss = self.model.train_step(enc_emb, dec_emb, self.optimizer, self.loss)
                self.log_training_loss(float(loss), epoch * len(pairs) + index)
                self.log_parameter_gradients(epoch * len(pairs) + index)
                loss_this_interval += loss
                if index % self.record_interval == 0:
                    avg_loss = loss_this_interval / self.record_interval
                    loss_this_interval = 0
                    logger.info("------------------------------------")
                    logger.info(f"{index} iteration mean loss = {avg_loss}")
                    logger.info(f"Time this chunk: {time.time() - prev_chunk_start_time}")
                    prev_chunk_start_time = time.time()
                    loss_to_plot.append(avg_loss)
                    self._log_examples_this_interval(pairs[:10])
                    self.tb_writer.flush()
            if validation_pairs is not None:
                logger.info("Checking for early stopping!")
                # Add check for minimum number of passes
                if self.early_stopping_criterion_met(validation_pairs):
                    break
            self.scheduler.step()
            logger.info("============================================")
        logger.info("----------")
        logger.info(f"Training took {(time.time() - start_time) / 60} minutes")
        self.tb_writer.close()
        return loss_to_plot

    def sample_generations_and_references(self, pairs) -> Tuple[List[str], List[str]]:
        best_outputs = []
        ref_outputs = []
        for in_indices, out_indices in pairs:
            # TGen does beam_size 10 and sets expansion size to be the same
            # (See TGen config.yaml line 35 and seq2seq.py line 219 `new_paths.extend(path.expand(self.beam_size, out_probs, st))`)
            cur_outputs = self.model.generate_beam(in_indices, beam_size=10, num_expansions=10)
            # The best output is the first one in the list, and the list contains pairs of length normalised logprobs along with the output indices
            best_outputs.append(self.output_vocab.pretty_string(cur_outputs[0][1]))
            ref_outputs.append(self.output_vocab.pretty_string(out_indices.tolist()))
        return best_outputs, ref_outputs

    def early_stopping_criterion_met(self, validation_pairs) -> bool:
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


class TGenTrainer(Seq2SeqAttnTrainer):
    def __init__(self,
                 model: "enunlg.encdec.tgen.TGenEncDec",
                 training_config=None):
        if training_config is None:
            # Set defaults
            training_config = omegaconf.DictConfig({"num_epochs": 20,
                                                    "record_interval": 1000,
                                                    "shuffle": True,
                                                    "batch_size": 1,
                                                    "optimizer": "adam",
                                                    "learning_rate": 0.0005,
                                                    "learning_rate_decay": 0.5  # TGen used 0.0
                                                   })
        super().__init__(model, training_config, input_vocab=model.input_vocab, output_vocab=model.output_vocab)


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
