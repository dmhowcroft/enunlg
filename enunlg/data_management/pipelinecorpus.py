from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union

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