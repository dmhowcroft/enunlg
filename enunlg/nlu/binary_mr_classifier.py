from typing import TYPE_CHECKING

import logging
import os
import tarfile
import tempfile

import omegaconf
import torch
import torch.nn

import enunlg.encdec.tgen

if TYPE_CHECKING:
    import enunlg.embeddings.binary
    import enunlg.vocabulary

logger = logging.getLogger(__name__)


class TGenSemClassifier(torch.nn.Module):
    def __init__(self, text_vocab_size: int,
                 bitvector_encoder_dims: int,
                 model_config=None) -> None:
        super().__init__()
        if model_config is None:
            # Set defaults
            model_config = omegaconf.DictConfig({'name': 'tgen_classifier',
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

        self.text_vocab_size = text_vocab_size
        self.bitvector_encoder_dims = bitvector_encoder_dims

        self.text_encoder = enunlg.encdec.tgen.TGenEnc(self.text_vocab_size, self.num_hidden_dims)
        self.classif_linear = torch.nn.Linear(self.num_hidden_dims, self.bitvector_encoder_dims)
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
        with open(f"{filepath}/_init_args.yaml", 'w') as init_args_file:
            omegaconf.OmegaConf.save({'text_vocab_size': self.text_vocab_size,
                                      'bitvector_encoder_dims': self.bitvector_encoder_dims},
                                     init_args_file)
        if tgz:
            with tarfile.open(f"{filepath}.tgz", mode="x:gz") as out_file:
                out_file.add(filepath, arcname=os.path.basename(filepath))


    @classmethod
    def load(cls, filepath):
        if tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, 'r') as classifier_file:
                tmp_dir = tempfile.mkdtemp()
                tarfile_member_names = classifier_file.getmembers()
                classifier_file.extractall(tmp_dir)
                root_name = tarfile_member_names[0].name
                with open(os.path.join(tmp_dir, root_name, "__class__.__name__"), 'r') as class_name_file:
                    class_name = class_name_file.read().strip()
                    assert class_name == cls.__name__
                init_args = omegaconf.OmegaConf.load(os.path.join(tmp_dir, root_name, '_init_args.yaml'))
                model_config = omegaconf.OmegaConf.load(os.path.join(tmp_dir, root_name, 'model_config.yaml'))
                model = cls(init_args.text_vocab_size, init_args.bitvector_encoder_dims, model_config)
                state_dict = torch.load(os.path.join(tmp_dir, root_name, '_state_dict.pt'))
                model.load_state_dict(state_dict)
                return model
