from typing import List, TYPE_CHECKING

import logging
import os
import random
import time

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

    @property
    def num_hidden_dims(self):
        return self.config.text_encoder.num_hidden_dims

    def forward(self, input_text_ints):
        enc_h_c_state = self.text_encoder.initial_h_c_state()
        enc_outputs, _ = self.text_encoder(input_text_ints, enc_h_c_state)
        output = self.classif_linear(enc_outputs.squeeze(0)[-1])
        output = self.classif_sigmoid(output)
        return output

    def train_step(self, text_ints, mr_onehot, optimizer, criterion):
        optimizer.zero_grad()

        output = self.forward(text_ints)
        logger.debug(f"{mr_onehot.size()=}")
        logger.debug(f"{output.size()=}")
        loss = criterion(output, mr_onehot)

        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, text_ints):
        with torch.no_grad():
            return self.forward(text_ints)

    def _save_classname_to_dir(self, directory_path):
        with open(os.path.join(directory_path, "__class__.__name__"), 'w') as class_file:
            class_file.write(self.__class__.__name__)

    def save(self, filepath, tgz=True):
        os.mkdir(filepath)
        self._save_classname_to_dir(filepath)
        with open(f"{filepath}/_state_dict.pt", 'wb') as state_file:
            torch.save(self.state_dict(), state_file)
        with open(f"{filepath}/model_config.yaml", 'w') as config_file:
            omegaconf.OmegaConf.save(self.config, config_file)
        # Warning bc we've written this model to depend on vocabs which we don't necessarily need as init args
        logger.warning("init args cannot be saved for this model yet")
        if tgz:
            with tarfile.open(f"{filepath}.tgz", mode="x:gz") as out_file:
                out_file.add(filepath, arcname=os.path.basename(filepath))