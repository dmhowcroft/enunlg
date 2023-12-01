from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import logging
import random

import enunlg.data_management.iocorpus

logger = logging.getLogger(__name__)


def string_to_python_identifier(text: str):
    return text.replace("-", "_").replace(" ", "_").strip()


class PipelineItem(object):
    def __init__(self, annotation_layers: Dict[str, Any]):
        """
        An entry in a pipeline dataset containing all the annotation_layers for a single example from the corpus.

        Each annotation layer with name `layer_name` for PipelineItem `x` can be accessed as x.layer_name or x['layer_name'].
        This is why layer names must be valid Python identifiers.

        :param annotation_layers: dict mapping layer names to the entry for that layer
        """
        self.annotation_layers = [string_to_python_identifier(layer_name) for layer_name in annotation_layers.keys()]
        for new_name, layer in zip(self.annotation_layers, annotation_layers):
            self.__setattr__(new_name, annotation_layers[layer])

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __repr__(self):
        attr_string = ", ".join([f"{layer}={str(self[layer])}" for layer in self.annotation_layers])
        return f"{self.__class__.__name__}({attr_string})"

    def print_layers(self):
        for layer in self.annotation_layers:
            layer_content = " ".join(self[layer])
            print(f"{layer}|\t{layer_content}")


AnyPipelineItemSubclass = TypeVar("AnyPipelineItemSubclass", bound=PipelineItem)


class PipelineCorpus(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[List[AnyPipelineItemSubclass]] = None, metadata: Optional[dict] = None):
        """Each item in a PipelineCorpus is a single entry with annotations for each stage of the pipeline."""
        if metadata is None:
            self.metadata = {'name': None,
                             'splits': None,
                             'directory': None,
                             'annotation_layers': None
                             }
        if seq:
            layer_names = seq[0].annotation_layers
            assert all([item.annotation_layers == layer_names for item in seq]), f"Expected all items in seq to have the layers: {layer_names}"
            self.annotation_layers = layer_names
        super(PipelineCorpus, self).__init__(seq)

    @property
    def layer_pairs(self):
        """Layers are listed in order, so adjacent pairs of annotation layers form individual Pipeline subtasks."""
        return [(l1, l2) for l1, l2 in zip(self.annotation_layers, self.annotation_layers[1:])]

    def items_by_layer_pair(self, layer_pair: Tuple[str, str]):
        layer_from, layer_to = layer_pair
        for item in self:
            yield item[layer_from], item[layer_to]

    def items_by_layer(self, layer_name):
        for item in self:
            yield item[layer_name]

    def print_summary_stats(self):
        print(f"{self.metadata=}")
        print(", ".join(self.annotation_layers))
        print(f"num entries: {len(self)}")
        num_entries_per_layer = defaultdict(int)
        layer_stats = defaultdict(int)
        layer_types = defaultdict(set)
        for layer in self.annotation_layers:
            for entry in self:
                layer_stats[layer] += len(entry[layer])
                layer_types[layer].update(entry[layer])
                num_entries_per_layer[layer] += 1
        for layer in layer_stats:
            print(f"{layer}:\t\t{layer_stats[layer] / num_entries_per_layer[layer]} ({num_entries_per_layer[layer]})")
            print(f"    with {len(layer_types[layer])} unique tokens.")

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


class TextPipelineCorpus(PipelineCorpus):
    def __init__(self, seq: Optional[List[AnyPipelineItemSubclass]] = None, metadata: Optional[dict] = None):
        super(TextPipelineCorpus, self).__init__(seq, metadata)
        self._max_layer_length = -1
        self._layer_lengths = {layer_name: -1 for layer_name in self.annotation_layers}

    @classmethod
    def from_existing(cls, corpus: PipelineCorpus, mapping_functions):
        out_corpus = TextPipelineCorpus(corpus)
        for item in out_corpus:
            for layer in item.annotation_layers:
                item[layer] = mapping_functions[layer](item[layer])
        return out_corpus

    @property
    def max_layer_length(self) -> int:
        if self._max_layer_length == -1:
            for item in self:
                for layer in item.annotation_layers:
                    if len(item[layer]) > self._max_layer_length:
                        logging.debug(f"New longest field, this time a {layer}")
                        logging.debug(item[layer])
                        self._max_layer_length = len(item[layer])
                    if len(item[layer]) > self._layer_lengths[layer]:
                        self._layer_lengths[layer] = len(item[layer])
        return self._max_layer_length

    def layer_length(self, layer_name: str) -> int:
        return self._layer_lengths[layer_name]
