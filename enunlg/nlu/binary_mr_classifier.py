from typing import List, TYPE_CHECKING

import logging
import random
import time

import numpy as np
import omegaconf
import torch
import torch.nn

import enunlg.encdec.tgen

if TYPE_CHECKING:
    import enunlg.embeddings.binary
    import enunlg.vocabulary

logger = logging.getLogger(__name__)


class TGenSemClassifier(torch.nn.Module):
    def __init__(self, text_vocabulary: "enunlg.vocabulary.TokenVocabulary",
                 onehot_encoder: "enunlg.embeddings.binary.DialogueActEmbeddings",
                 model_config=None) -> None:
        super().__init__()
        if model_config is None:
            # Set defaults
            model_config = omegaconf.DictConfig({'name': 'tgen_classifier',
                                    'max_mr_length': 30,
                                    'text_encoder':
                                        {'embeddings':
                                            {'mode': 'random',
                                             'dimensions': 50,
                                             'backprop': True
                                             },
                                         'cell': 'lstm',
                                         'num_hidden_dims': 128}
                                    })
        self.config = model_config

        self.text_vocabulary = text_vocabulary
        self.onehot_encoder = onehot_encoder

        self.text_encoder = enunlg.encdec.tgen.TGenEnc(self.text_vocabulary.max_index + 1, self.num_hidden_dims)
        self.classif_linear = torch.nn.Linear(self.num_hidden_dims, self.onehot_encoder.dimensionality)
        self.classif_sigmoid = torch.nn.Sigmoid()

        # Initialise optimisers (same as in TGenEncDec model)
        self.learning_rate = 0.0005
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

    @property
    def num_hidden_dims(self):
        return self.config.text_encoder.num_hidden_dims

    def forward(self, input_text_ints):
        enc_h_c_state = self.text_encoder.initial_h_c_state()
        enc_outputs, _ = self.text_encoder(input_text_ints, enc_h_c_state)
        output = self.classif_linear(enc_outputs.squeeze(0)[-1])
        output = self.classif_sigmoid(output)
        return output

    def train_step(self, text_ints, mr_onehot):
        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()

        output = self.forward(text_ints)
        logger.debug(f"{mr_onehot.size()=}")
        logger.debug(f"{output.size()=}")
        loss = criterion(output, mr_onehot)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_iterations(self, pairs, epochs: int, record_interval: int = 1000) -> List[float]:
        """
        Run `epochs` training epochs over all training pairs, shuffling pairs in place each epoch.

        :param pairs: input and output indices for embeddings
        :param epochs: number of training epochs to run
        :param record_interval: how frequently to print and record training loss
        :return: list of average loss for each `record_interval` for each epoch
        """
        # In TGen this would begin with a call to self._init_training()

        start_time = time.time()
        prev_chunk_start_time = start_time
        loss_this_interval = 0
        loss_to_plot = []

        for epoch in range(epochs):
            logger.info(f"Beginning epoch {epoch}...")
            logger.info(f"Learning rate is now {self.learning_rate}")
            random.shuffle(pairs)
            for index, (text_ints, mr_onehot) in enumerate(pairs, start=1):
                loss = self.train_step(text_ints, mr_onehot)
                loss_this_interval += loss
                if index % record_interval == 0:
                    avg_loss = loss_this_interval / record_interval
                    loss_this_interval = 0
                    logger.info("------------------------------------")
                    logger.info(f"{index} iteration mean loss = {avg_loss}")
                    logger.info(f"Time this chunk: {time.time() - prev_chunk_start_time}")
                    prev_chunk_start_time = time.time()
                    loss_to_plot.append(avg_loss)
                    for i, o in pairs[:10]:
                        logger.info("An example!")
                        logger.info(f"Text:   {' '.join([x for x in self.text_vocabulary.get_tokens(i.tolist()) if x != '<VOID>'])}")
                        logger.info(f"MR:     {self.onehot_encoder.embedding_to_string(o.tolist())}")
                        prediction = self.predict(i).squeeze(0).squeeze(0).tolist()
                        # output_list = [1.0 if x > 0.95 else 0.0 for x in prediction]
                        # logger.info(f"Output: {self.onehot_encoder.embedding_to_string(output_list)}")
                        logger.info(f"Output: {self.onehot_encoder.embedding_to_string(list(np.round(prediction)))}")
                        target_bitvector = np.round(o.tolist())
                        output_bitvector = np.round(prediction)
                        logger.info(f"Target bitvector: {target_bitvector}")
                        logger.info(f"Output bitvector: {output_bitvector}")
                        logger.info(f"Error: {sum(abs(target_bitvector - output_bitvector))}")
            self.scheduler.step()
            logger.info("============================================")
        logger.info("----------")
        logger.info(f"Training took {(time.time() - start_time) / 60} minutes")
        return loss_to_plot

    def predict(self, text_ints):
        with torch.no_grad():
            return self.forward(text_ints)