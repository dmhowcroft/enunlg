import os
import tarfile
import tempfile
from typing import List, Dict, Optional

import omegaconf
import torch

import enunlg.data_management
import enunlg.encdec.multitask_seq2seq
import enunlg.vocabulary


class MultitaskSeq2SeqGenerator(object):
    STATE_ATTRIBUTES = ('layers', 'vocabularies', 'max_length_any_layer', 'corpus_metadata', 'model')

    def __init__(self, corpus: enunlg.data_management.pipelinecorpus.TextPipelineCorpus, model_config: omegaconf.DictConfig):
        """
        Create a multi-decoder seq2seq+attn model based on `corpus`.

        The first layer will be treated as input, subsequent layers will be treated as targets for decoding.
        At training time we use all the decoding layers, but at inference time we only decode at the final layer.

        :param corpus:
        """
        self.layers: List[str] = corpus.annotation_layers
        self.vocabularies: Dict[str, enunlg.vocabulary.TokenVocabulary] = {layer: enunlg.vocabulary.TokenVocabulary(list(corpus.items_by_layer(layer))) for layer in self.layers} # type: ignore[misc]
        # Store some basic information about the corpus
        self.max_length_any_layer = corpus.max_layer_length
        self.corpus_metadata = corpus.metadata
        self.model = enunlg.encdec.multitask_seq2seq.DeepEncoderMultiDecoderSeq2SeqAttn(self.layers, [self.vocabularies[layer].size for layer in self.vocabularies], model_config)

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

    def _save_classname_to_dir(self, directory_path):
        with open(os.path.join(directory_path, "__class__.__name__"), 'w') as class_file:
            class_file.write(self.__class__.__name__)

    def save(self, filepath, tgz=True):
        os.mkdir(filepath)
        self._save_classname_to_dir(filepath)
        state = {}
        for attribute in self.STATE_ATTRIBUTES:
            curr_obj = getattr(self, attribute)
            save_method = getattr(curr_obj, 'save', None)
            print(curr_obj)
            print(save_method)
            if attribute == "vocabularies":
                # Special handling for the vocabularies
                state[attribute] = {}
                os.mkdir(f"{filepath}/{attribute}")
                for key in curr_obj:
                    state[attribute][key] = f"./{attribute}/{key}"
                    curr_obj[key].save(f"{filepath}/{attribute}/{key}", tgz=False)
            elif save_method is None:
                state[attribute] = curr_obj
            else:
                state[attribute] = f"./{attribute}"
                curr_obj.save(f"{filepath}/{attribute}", tgz=False)
        with open(os.path.join(filepath, "_save_state.yaml"), 'w') as state_file:
            state = omegaconf.OmegaConf.create(state)
            omegaconf.OmegaConf.save(state, state_file)
        if tgz:
            with tarfile.open(f"{filepath}.tgz", mode="x:gz") as out_file:
                out_file.add(filepath, arcname=os.path.basename(filepath))

    @classmethod
    def load(cls, filepath):
        if tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, 'r') as generator_file:
                tmp_dir = tempfile.mkdtemp()
                tarfile_member_names = generator_file.getmembers()
                generator_file.extractall(tmp_dir)
                root_name = tarfile_member_names[0].name
                with open(os.path.join(tmp_dir, root_name, "__class__.__name__"), 'r') as class_name_file:
                    class_name = class_name_file.read().strip()
                    assert class_name == cls.__name__
                model = enunlg.encdec.multitask_seq2seq.DeepEncoderMultiDecoderSeq2SeqAttn.load_from_dir(os.path.join(tmp_dir, root_name, 'model'))
                dummy_pipeline_item = enunlg.data_management.pipelinecorpus.PipelineItem({layer_name: "" for layer_name in model.layer_names})
                dummy_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus([dummy_pipeline_item])
                dummy_corpus.pop()
                new_generator = cls(dummy_corpus, model.config)
                new_generator.model = model
                state_dict = omegaconf.OmegaConf.load(os.path.join(tmp_dir, root_name, "_save_state.yaml"))
                vocabs = {}
                for vocab in state_dict.vocabularies:
                    vocabs[vocab] = enunlg.vocabulary.TokenVocabulary.load_from_dir(os.path.join(tmp_dir, root_name, 'vocabularies', vocab))
                new_generator.vocabularies = vocabs
                return new_generator


    def prep_embeddings(self, corpus, max_length: Optional[int]=None):
        layer_names = list(self.vocabularies.keys())
        input_layer_name = layer_names[0]
        if max_length is None:
            max_length = corpus.max_layer_length
        input_embeddings = [torch.tensor(self.vocabularies[input_layer_name].get_ints_with_left_padding(item, max_length),
                                         dtype=torch.long) for item in corpus.items_by_layer(input_layer_name)]
        output_embeddings = {
            layer_name: [torch.tensor(self.vocabularies[layer_name].get_ints(item), dtype=torch.long) for item in
                    corpus.items_by_layer(layer_name)] for layer_name in layer_names[1:]}
        return input_embeddings, output_embeddings
