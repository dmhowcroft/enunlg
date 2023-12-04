from collections import defaultdict
from collections.abc import MutableMapping
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import difflib
import logging
import os

import omegaconf
import xsdata.formats.dataclass.parsers as xsparsers

from enunlg.formats.xml.enriched_e2e import EnrichedE2EEntries, EnrichedE2EEntry
from enunlg.meaning_representation.slot_value import SlotValueMR
from enunlg.normalisation.tokenisation import TGenTokeniser

import enunlg.data_management.pipelinecorpus

logger = logging.getLogger(__name__)

# TODO add hydra configuration for enriched e2e stuff
ENRICHED_E2E_CONFIG = omegaconf.DictConfig({'ENRICHED_E2E_DIR': os.path.join(os.path.dirname(__file__),
                                                                             '../../datasets/processed/EnrichedE2E/')})

E2E_SPLIT_DIRS = ('train', 'dev', 'test')

DELEX_LABELS = ["__AREA__", "__CUSTOMER_RATING__", "__EATTYPE__", "__FAMILYFRIENDLY__", "__FOOD__", "__NAME__",
                "__NEAR__", "__PRICERANGE__"]

DIFFER = difflib.Differ()


def extract_reg_from_template_and_text(text: str, template: str) -> MutableMapping[str, List[str]]:
    diff = DIFFER.compare(text.strip().split(), template.strip().split())
    keys = []
    values = []
    curr_key = []
    curr_value = []
    for line in diff:
        # print(line)
        if line.startswith('-'):
            curr_value.append(line.split()[1])
        elif line.startswith('+'):
            token = line.split()[1]
            curr_key.append(token)

        else:
            if curr_value:
                values.append(" ".join(curr_value))
                curr_value = []
            if curr_key:
                keys.append(" ".join(curr_key))
                curr_key = []
    # Add the last key & value to the dict if we have some
    if curr_value:
        values.append(" ".join(curr_value))
    if curr_key:
        keys.append(" ".join(curr_key))
    result_dict = defaultdict(list)
    for key, value in zip(keys, values):
        result_dict[key].append(value)
    return result_dict


class EnrichedE2ECorpusRaw(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[Iterable] = None, filename_or_list: Optional[Union[str, List[str]]] = None):
        super().__init__(seq)
        if filename_or_list is not None:
            if isinstance(filename_or_list, list):
                for filename in filename_or_list:
                    logger.info(filename)
                    self.load_file(filename)
            elif isinstance(filename_or_list, str):
                self.load_file(filename_or_list)
            else:
                raise TypeError(f"Expected filename_or_list to be None, str, or list, not {type(filename_or_list)}")

    def load_file(self, filename):
        entries_object = xsparsers.XmlParser().parse(filename, EnrichedE2EEntries)
        self.extend(entries_object.entries)


class PipelineCorpusMapper(object):
    def __init__(self, input_format, output_format, annotation_layer_mappings: Dict[str, Callable]):
        """
        Create a function which will map from `input_format` to `output_format` using `annotation_layer_mappings`.
        """
        self.input_format = input_format
        self.output_format = output_format
        self.annotation_layer_mappings = annotation_layer_mappings

    def __call__(self, input_corpus: Iterable) -> List:
        logger.debug(f'successful call to {self.__class__.__name__} as a function (rather than a class)')
        if isinstance(input_corpus, self.input_format):
            logger.debug('passed the format check')
            output_seq = []
            for entry in input_corpus:
                output = []
                for layer in self.annotation_layer_mappings:
                    logger.debug(f"processing {layer}")
                    output.append(self.annotation_layer_mappings[layer](entry))
                # EnrichedE2E-formated datasets have up to N distinct targets for each single input
                # This will show up as the first 'layer' having length 1 and subsequent layers having length > 1
                num_targets = max([len(x) for x in output])
                # We expand any layers of length 1, duplicating their entries, and preserving the rest of the layers
                output = [x * num_targets if len(x) == 1 else x for x in output]
                assert all([len(x) == num_targets for x in output]), f"expected all layers to have the same number of items, but received: {[len(x) for x in output]}"
                # For each of the N distinct targets, create a self.outputformat object and append it to the output_seq
                for i in range(num_targets-1):
                    item = self.output_format({key: output[idx][i] for idx, key in enumerate(self.annotation_layer_mappings.keys())})
                    output_seq.append(item)
                logger.debug(f"Num entries so far: {len(output_seq)}")
            return output_seq
        else:
            raise TypeError(f"Cannot run {self.__class__} on {type(input_corpus)}")


class EnrichedE2EItem(enunlg.data_management.pipelinecorpus.PipelineItem):
    def __init__(self, annotation_layers):
        super().__init__(annotation_layers)


class EnrichedE2ECorpus(enunlg.data_management.pipelinecorpus.PipelineCorpus):
    def __init__(self, seq: List[EnrichedE2EItem], metadata=None):
        super(EnrichedE2ECorpus, self).__init__(seq, metadata)


def extract_raw_input(entry: EnrichedE2EEntry) -> List[SlotValueMR]:
    mr = {}
    for source_input in entry.source.inputs:
        mr[source_input.attribute] = source_input.value
    return [SlotValueMR(mr, frozen_box=True)]


def extract_selected_input(entry: EnrichedE2EEntry) -> List[SlotValueMR]:
    targets = []
    for target in entry.targets:
        mr = {}
        for sentence in target.structuring.sentences:
            for input in sentence.content:
                mr[input.attribute] = input.value
        targets.append(SlotValueMR(mr, frozen_box=True))
    return targets


def extract_ordered_input(entry: EnrichedE2EEntry) -> List[SlotValueMR]:
    targets = []
    for target in entry.targets:
        mr = {}
        for sentence in target.structuring.sentences:
            for input in sentence.content:
                mr[input.attribute] = input.value
        targets.append(SlotValueMR(mr, frozen_box=True))
    return targets


def extract_sentence_segmented_input(entry: EnrichedE2EEntry) -> List[Tuple[SlotValueMR]]:
    targets = []
    for target in entry.targets:
        selected_inputs = []
        for sentence in target.structuring.sentences:
            mr = {}
            for input in sentence.content:
                mr[input.attribute] = input.value
            selected_inputs.append(SlotValueMR(mr, frozen_box=True))
        targets.append(tuple(selected_inputs))
    return targets


def extract_lexicalization(entry: EnrichedE2EEntry) -> List[str]:
    return [target.lexicalization for target in entry.targets]


def extract_reg_in_lex(entry: EnrichedE2EEntry) -> List[str]:
    texts = [target.text for target in entry.targets]
    templates = [target.template for target in entry.targets]
    lexes = [target.lexicalization for target in entry.targets]
    reg_lexes = []
    for text, template, lex in zip(texts, templates, lexes):
        reg_dict = extract_reg_from_template_and_text(text, template)
        new_lex = []
        curr_text_idx = 0
        for lex_token in lex.split():
            if lex_token.startswith("__"):
                # print(f"looking for {lex_token}")
                possible_targets = reg_dict[lex_token]
                match_found = False
                curr_rest = text.split()[curr_text_idx:]
                for possible_target in possible_targets:
                    target_tokens = tuple(possible_target.split())
                    num_tokens = len(target_tokens)
                    # print(target_tokens)
                    # print(curr_rest)
                    if num_tokens == 1:
                        for text_idx, text_token in enumerate(curr_rest):
                            if text_token in possible_targets:
                                new_lex.append(text_token)
                                curr_text_idx += text_idx
                                match_found = True
                                break
                    elif num_tokens > 1:
                        parts = []
                        for i in range(len(target_tokens)):
                            parts.append(curr_rest[i:])
                        for start_idx, token_tuple in enumerate(zip(*parts)):
                            # print(token_tuple)
                            if token_tuple == target_tokens:
                                new_lex.extend(target_tokens)
                                curr_text_idx += start_idx + len(target_tokens)
                                match_found = True
                                break
                    else:
                        raise ValueError("Must have possible targets for each slot!")
                    if match_found:
                        break
                if not match_found:
                    print(f"could not find: {lex_token}")
                    new_lex.append(lex_token)
                    # raise ValueError(f"Could not create reg_lex text for {lex_token} in:\n{text}\n{template}\n{lex}\n{reg_dict}")
            else:
                new_lex.append(lex_token)
        reg_lexes.append(" ".join(new_lex))
    return reg_lexes


def extract_raw_output(entry: EnrichedE2EEntry) -> List[str]:
    return [target.text for target in entry.targets]


def load_enriched_e2e(splits: Optional[Iterable[str]] = None, enriched_e2e_config: Optional[omegaconf.DictConfig] = None) -> EnrichedE2ECorpus:
    """
    :param splits: which splits to load
    :param enriched_e2e_config: a SlotValueMR or omegaconf.DictConfig like object containing the basic
                                information about the e2e corpus to be used
    :return: the corpus of MR-text pairs with metadata
    """
    if enriched_e2e_config is None:
        enriched_e2e_config = ENRICHED_E2E_CONFIG
    corpus_name = "Enriched E2E Challenge Corpus"
    default_splits = E2E_SPLIT_DIRS
    data_directory = enriched_e2e_config.ENRICHED_E2E_DIR
    if splits is None:
        splits = default_splits
    elif not set(splits).issubset(default_splits):
        raise ValueError(f"`splits` can only contain a subset of {default_splits}. Found {splits}.")
    fns = []
    for split in splits:
        logger.info(split)
        fns.extend([os.path.join(data_directory, split, fn) for fn in os.listdir(os.path.join(data_directory, split))])

    corpus: EnrichedE2ECorpusRaw = EnrichedE2ECorpusRaw(filename_or_list=fns)
    corpus.metadata = {'name': corpus_name,
                       'splits': splits,
                       'directory': data_directory}
    logger.info(len(corpus))

    # tokenize texts
    for entry in corpus:
        for target in entry.targets:
            target.text = TGenTokeniser.tokenise(target.text)
    enriched_e2e_factory = PipelineCorpusMapper(EnrichedE2ECorpusRaw, EnrichedE2EItem,
                                                {'raw-input': lambda entry: extract_raw_input(entry),
                                                 'selected-input': extract_selected_input,
                                                 'ordered-input': extract_ordered_input,
                                                 'sentence-segmented-input': extract_sentence_segmented_input,
                                                 'lexicalisation': extract_lexicalization,
                                                 'referring-expressions': extract_reg_in_lex,
                                                 'raw-output': extract_raw_output})

    # Specify the type again since we're changing the expected type of the variable and mypy doesn't like that
    corpus: EnrichedE2ECorpus = EnrichedE2ECorpus(enriched_e2e_factory(corpus))
    logger.info(f"Corpus contains {len(corpus)} entries.")
    return corpus


def sanitize_values(value):
    return value.replace(" ", "_").replace("'", "_")


def sanitize_slot_names(slot_name):
    return slot_name


def linearize_slot_value_mr(mr: enunlg.meaning_representation.slot_value.SlotValueMR):
    tokens = ["<MR>"]
    for slot in mr:
        tokens.append(sanitize_slot_names(slot))
        tokens.append(sanitize_values(mr[slot]))
        tokens.append("<PAIR_SEP>")
    tokens.append("</MR>")
    return tokens


def linearize_slot_value_mr_seq(mrs):
    tokens = ["<SENTENCE>"]
    for mr in mrs:
        tokens.extend(linearize_slot_value_mr(mr))
        tokens.append("</SENTENCE>")
    return tokens
