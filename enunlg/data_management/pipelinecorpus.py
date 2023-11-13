from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union

import logging
import random

import enunlg.data_management.iocorpus


def string_to_python_identifier(text: str):
    return text.replace("-", "_").replace(" ", "_").strip()


class PipelineItem(object):
    def __init__(self, layers: Dict[str, Any]):
        """
        :param layers: dict mapping layer names to the entry for that layer
        """
        self.layers = [string_to_python_identifier(layer_name) for layer_name in layers.keys()]
        for new_name, layer in zip(self.layers, layers):
            self.__setattr__(new_name, layers[layer])

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __repr__(self):
        attr_string = ", ".join([f"{layer}={str(self[layer])}" for layer in self.layers])
        return f"{self.__class__.__name__}({attr_string})"

    def print_layers(self):
        for layer in self.layers:
            layer_content = " ".join(self[layer])
            print(f"{layer}|\t{layer_content}")


AnyPipelineItemSubclass = TypeVar("AnyPipelineItemSubclass", bound=PipelineItem)


class PipelineCorpus(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[List[AnyPipelineItemSubclass]] = None, metadata: Optional[dict] = None):
        if metadata is None:
            self.metadata = {'name': None,
                             'splits': None,
                             'directory': None,
                             'annotation_layers': None
                             }
        if seq:
            layer_names = seq[0].layers
            assert all([item.layers == layer_names for item in seq]), f"Expected all items in seq to have the layers: {layer_names}"
            self.layers = layer_names
        super(PipelineCorpus, self).__init__(seq)

    @property
    def layer_pairs(self):
        return [(l1, l2) for l1, l2 in zip(self.layers, self.layers[1:])]

    def items_by_layer_pair(self, layer_pair):
        for item in self:
            yield item[layer_pair[0]], item[layer_pair[1]]

    def items_by_layer(self, layer_name):
        for item in self:
            yield item[layer_name]

    def print_summary_stats(self):
        print(f"{self.metadata=}")
        print(", ".join(self.layers))
        print(f"num entries: {len(self)}")
        num_entries_per_layer = defaultdict(int)
        layer_stats = defaultdict(int)
        for layer in self.layers:
            for entry in self:
                layer_stats[layer] += len(entry[layer])
                num_entries_per_layer[layer] += 1
        for layer in layer_stats:
            print(f"{layer}:\t\t{layer_stats[layer] / num_entries_per_layer[layer]} ({num_entries_per_layer[layer]})")

    def print_sample(self, range_start=0, range_end=10, subsample=None):
        if random is None:
            for item in self[range_start:range_end]:
                item.print_layers()
                print("----")
        elif isinstance(subsample, int):
            for item in random.choices(self[range_start:range_end], k=subsample):
                item.print_layers()
                print("----")
        else:
            raise ValueError("`random` must be None or an integer")


class TextPipelineCorpus(enunlg.data_management.pipelinecorpus.PipelineCorpus):
    def __init__(self, seq: Optional[List[AnyPipelineItemSubclass]] = None, metadata: Optional[dict] = None):
        super(TextPipelineCorpus, self).__init__(seq, metadata)
        self._max_layer_length = -1
        self._layer_lengths = {layer_name: -1 for layer_name in self.layers}

    @classmethod
    def from_existing(cls, corpus: enunlg.data_management.pipelinecorpus.PipelineCorpus, mapping_functions):
        out_corpus = TextPipelineCorpus(corpus)
        for item in out_corpus:
            for layer in item.layers:
                item[layer] = mapping_functions[layer](item[layer])
        return out_corpus

    @property
    def max_layer_length(self):
        if self._max_layer_length == -1:
            for item in self:
                for layer in item.layers:
                    if len(item[layer]) > self._max_layer_length:
                        logging.debug(f"New longest field, this time a {layer}")
                        logging.debug(item[layer])
                        self._max_layer_length = len(item[layer])
                    if len(item[layer]) > self._layer_lengths[layer]:
                        self._layer_lengths[layer] = len(item[layer])
        return self._max_layer_length
