from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO, Tuple, TypeVar, Union

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
    STATE_ATTRIBUTES = ("metadata", "annotation_layers")
    def __init__(self, seq: Optional[List[AnyPipelineItemSubclass]] = None, metadata: Optional[dict] = None):
        """Each item in a PipelineCorpus is a single entry with annotations for each stage of the pipeline."""
        if metadata is None:
            self.metadata = {'name': None,
                             'splits': None,
                             'directory': None
                             }
        if seq:
            layer_names = seq[0].annotation_layers
            assert all([item.annotation_layers == layer_names for item in seq]), f"Expected all items in seq to have the layers: {layer_names}"
            self.annotation_layers = layer_names
        super(PipelineCorpus, self).__init__(seq)

    def __getstate__(self):
        state = {attribute: self.__getattribute__(attribute)
                 for attribute in self.STATE_ATTRIBUTES}
        state['__class__'] = self.__class__.__name__
        state['_content'] = [x for x in self]
        return state

    @classmethod
    def __setstate__(cls, state: Dict[str, Any]):
        class_name = state["__class__"]
        assert class_name == cls.__name__
        new_generator = cls.__new__(cls)
        for attribute in cls.STATE_ATTRIBUTES:
            new_generator.__setattr__(attribute, state[attribute])
        new_generator.append(state['_content'])
        return new_generator

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
        layer_lengths = defaultdict(list)
        layer_types = defaultdict(set)
        for layer in self.annotation_layers:
            for entry in self:
                if entry[layer] is not None:
                    layer_lengths[layer].append(len(entry[layer]))
                    layer_types[layer].update(entry[layer])
                    num_entries_per_layer[layer] += 1
                else:
                    print("None type found for the entry {}".format(entry))
        for layer in layer_lengths:
            print(f"{layer}:\t\t{sum(layer_lengths[layer])/num_entries_per_layer[layer]:.2f} [{min(layer_lengths[layer])},{max(layer_lengths[layer])}]")
            print(f"    with {len(layer_types[layer])} types across {sum(layer_lengths[layer])} tokens.")

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
    STATE_ATTRIBUTES = tuple(list(PipelineCorpus.STATE_ATTRIBUTES) + ["_max_layer_length", "_layer_lengths"])

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
                        logger.debug(f"New longest field, this time a {layer}")
                        logger.debug(item[layer])
                        self._max_layer_length = len(item[layer])
                    if len(item[layer]) > self._layer_lengths[layer]:
                        self._layer_lengths[layer] = len(item[layer])
        return self._max_layer_length

    def layer_length(self, layer_name: str) -> int:
        return self._layer_lengths[layer_name]

    def all_item_layer_iterator(self):
        for item in self:
            for layer in self.annotation_layers:
                yield item[layer]

    def save(self, filename: str) -> None:
        with open(filename, 'w') as out_file:
            self.write_to_iostream(out_file)
    
    def write_to_iostream(self, io_stream: TextIO) -> None:
        io_stream.write("# TextPipeline Corpus Save File\n")
        io_stream.write("# Format Version 0.1\n")
        io_stream.write("\n")
        io_stream.write("# Annotation Layers:\n")
        for annotation_layer in self.annotation_layers:
            io_stream.write(f"#   {annotation_layer}\n")
        io_stream.write("\n")
        for entry in self:
            for annotation_layer in self.annotation_layers:
                layer_line = " ".join(entry[annotation_layer])
                io_stream.write(f"{layer_line}\n")
            io_stream.write("\n")


class PipelineCorpusMapper(object):
    def __init__(self, input_format, output_format, annotation_layer_mappings: Dict[str, Callable]):
        """
        Create a function which will map from `input_format` to `output_format` using `annotation_layer_mappings`.
        """
        self.input_format = input_format
        self.output_format = output_format
        self.annotation_layer_mappings = annotation_layer_mappings

    def __call__(self, input_corpus: Iterable) -> List:
        # logger.debug(f'successful call to {self.__class__.__name__} as a function (rather than a class)')
        if isinstance(input_corpus, self.input_format):
            # logger.debug('passed the format check')
            output_seq = []
            for entry in input_corpus:
                # Each entry can actually contain multiple lexicalisations,
                # so we need to build up the entry as we iterate.
                output = []
                for layer in self.annotation_layer_mappings:
                    # logger.debug(f"processing {layer}")
                    output.append(self.annotation_layer_mappings[layer](entry))
                # EnrichedWebNLG-formatted datasets have up to N distinct targets for each single input
                # This will show up as the first 'layer' having length 1 and subsequent layers having length > 1
                num_targets = max([len(x) for x in output])
                # We expand any layers of length 1, duplicating their entries, and preserving the rest of the layers
                output = [x * num_targets if len(x) == 1 else x for x in output]
                try:
                    assert all([len(x) == num_targets for x in output]), f"expected all layers to have the same number of items, but received: {[len(x) for x in output]}"
                except AssertionError:
                    print(entry)
                    for x in output:
                        print(x)
                    raise
                # For each of the N distinct targets, create a self.outputformat object and append it to the output_seq
                for i in range(num_targets-1):
                    item = self.output_format({key: output[idx][i] for idx, key in enumerate(self.annotation_layer_mappings.keys())})
                    output_seq.append(item)
                # logger.debug(f"Num entries so far: {len(output_seq)}")
            return output_seq
        else:
            raise TypeError(f"Cannot run {self.__class__} on {type(input_corpus)}")
