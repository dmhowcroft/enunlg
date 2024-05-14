from collections import defaultdict
from collections.abc import MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import difflib
import logging
import os

import omegaconf
import regex
import xsdata.exceptions
import xsdata.formats.dataclass.parsers as xsparsers

from enunlg.data_management.webnlg import RDFTriple, RDFTripleList
from enunlg.formats.xml.enriched_webnlg import EnrichedWebNLGBenchmark, EnrichedWebNLGEntry
from enunlg.normalisation.tokenisation import TGenTokeniser

import enunlg.data_management.pipelinecorpus

logger = logging.getLogger(__name__)

# TODO add hydra configuration for enriched e2e stuff
ENRICHED_WEBNLG_CONFIG = omegaconf.DictConfig({'ENRICHED_WEBNLG_DIR':
                                                   os.path.join(os.path.dirname(__file__),
                                                                '../../datasets/processed/webnlg/data/v1.6/en/')})

WEBNLG_SPLIT_DIRS = ('train', 'dev', 'test')

DELEX_LABELS = ["AGENT-1",
                "BRIDGE-1", "BRIDGE-2", "BRIDGE-3", "BRIDGE-4",
                "PATIENT-1", "PATIENT-2", "PATIENT-3", "PATIENT-4", "PATIENT-5", "PATIENT-6", "PATIENT-7"]


DIFFER = difflib.Differ()


def extract_reg_from_template_and_text(text: str, template: str, print_diff: bool = False) -> MutableMapping[str, List[str]]:
    diff = DIFFER.compare(text.strip().split(), template.strip().split())
    keys = []
    values = []
    curr_key = []
    curr_value = []
    for line in diff:
        if print_diff:
            logger.debug(line)
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


class EnrichedWebNLGCorpusRaw(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[Iterable] = None, filename_or_list: Optional[Union[str, List[str]]] = None):
        super().__init__(seq)
        if filename_or_list is not None:
            if isinstance(filename_or_list, list):
                for filename in filename_or_list:
                    logger.info(filename)
                    try:
                        self.load_file(filename)
                    except xsdata.exceptions.ParserError:
                        print(filename)
                        raise
            elif isinstance(filename_or_list, str):
                self.load_file(filename_or_list)
            else:
                message = f"Expected filename_or_list to be None, str, or list, not {type(filename_or_list)}"
                raise TypeError(message)

    def load_file(self, filename):
        benchmark_object = xsparsers.XmlParser().parse(filename, EnrichedWebNLGBenchmark)
        entries_object = benchmark_object.entries
        # TODO align names of XML objects like we did for E2E
        self.extend(entries_object.entry)


class EnrichedWebNLGReference(object):
    def __init__(self, entity, seq_loc, orig_delex_tag, ref_type, form):
        self.entity = str(entity)
        self.seq_loc = seq_loc
        self.orig_delex_tag = orig_delex_tag
        self.ref_type = ref_type
        self.form = form

class EnrichedWebNLGReferences(object):
    def __init__(self, ref_list):
        self.sequence = ref_list
        self.lookup_by_entity = defaultdict(list)
        self.entity_orig_tag_mapping = {}
        for index, ref in enumerate(self.sequence):
            self.lookup_by_entity[ref.entity].append(index)
            self.entity_orig_tag_mapping[ref.entity] = ref.orig_delex_tag


class EnrichedWebNLGItem(enunlg.data_management.pipelinecorpus.PipelineItem):
    def __init__(self, annotation_layers):
        """
        An EnrichedWebNLGItem contains `annotation_layers` corresponding do different stages of realisation
        as well as `references` encoding the references provided the Enriched WebNLG dataset.
        """
        super().__init__(annotation_layers)
        self.references = EnrichedWebNLGReferences([])
        self._delexicalization_tracking = []
        self._sem_class_counts = defaultdict(int)

    def delex_reference(self, entity, sem_class):
        # TODO delexicalisation should only succeed if *all* layers can be delexicalized w.r.t. that entity
        self._delexicalization_tracking.append((entity, sem_class))
        sem_class_string = f"__{sem_class}-{self._sem_class_counts[sem_class]}__"
        print(sem_class_string)
        orig_tag = self.references.entity_orig_tag_mapping[entity]
        print(orig_tag)
        for layer_name in self.annotation_layers:
            layer = self[layer_name]
            if isinstance(layer, RDFTripleList):
                layer.delex_reference(entity, sem_class_string)
            elif isinstance(layer, tuple):
                if all(isinstance(element, RDFTripleList) for element in layer):
                    for element in layer:
                        element.delex_reference(entity, sem_class_string)
            elif isinstance(layer, str):
                if layer_name == 'lexicalisation':
                    self[layer_name] = layer.replace(orig_tag, sem_class_string)
                else:
                    self[layer_name] = layer.replace(entity, sem_class_string).replace(entity.replace("_", " "), sem_class_string)
            else:
                raise ValueError(f"Unexpected type for this layer: {type(layer)}")
        print(self)
        self._sem_class_counts[sem_class] += 1

    def undo_enriched_webnlg_delex(self):
        for reference in self.references.sequence:
            # print(f"{reference.orig_delex_tag=}")
            self['lexicalisation'] = self['lexicalisation'].replace(reference.orig_delex_tag, str(reference.form), 1)


class EnrichedWebNLGCorpus(enunlg.data_management.pipelinecorpus.PipelineCorpus):
    def __init__(self, seq: List[EnrichedWebNLGItem], metadata=None):
        super(EnrichedWebNLGCorpus, self).__init__(seq, metadata)

    @classmethod
    def from_raw_corpus(cls, raw_corpus: EnrichedWebNLGCorpusRaw) -> "EnrichedWebNLGCorpus":
        """"""
        out_corpus = []
        for entry in raw_corpus:
            # One input with multiple 'lex' entries
            raw_input = extract_raw_input(entry)
            # Extract the 'layers' from this representation.
            # NB: switched to in-lining everything so it's all in one place and we don't have
            # to keep track of a bunch of different functions. Data munging is messy no matter where and how we do it.
            for lex in entry.lex:
                sentence_grouped_triples = []
                selected_ordered_triples = []
                for sentence in lex.sortedtripleset.sentence:
                    sentence_triples = []
                    for sortedtriple in sentence.striple:
                        this_triple = RDFTriple.from_string(sortedtriple)
                        selected_ordered_triples.append(this_triple)
                        sentence_triples.append(this_triple)
                    sentence_grouped_triples.append(tuple(sentence_triples))
                selected_input = RDFTripleList(selected_ordered_triples)
                ordered_input = RDFTripleList(selected_ordered_triples)
                sentence_segmented_input = tuple(sentence_grouped_triples)
                lexicalization = lex.lexicalization
                raw_output = lex.text
                # This will drop any entries which contain 'None' for any annotation layers
                if None in [raw_output, lex.template, lexicalization]:
                    continue
                if any([None in x for x in [raw_input, selected_input, ordered_input, sentence_segmented_input]]):
                    continue
                if any([len(x) == 0 for x in [raw_input, selected_input, ordered_input, sentence_segmented_input]]):
                    continue
                new_item = EnrichedWebNLGItem({'raw_input': deepcopy(raw_input),
                                               'selected_input': selected_input,
                                               'ordered_input': ordered_input,
                                               'sentence_segmented_input': sentence_segmented_input,
                                               'lexicalisation': lexicalization,
                                               'raw_output': raw_output})
                new_item.references = extract_refs_from_xsdata_rep(lex.references.reference)
                out_corpus.append(new_item)
        return cls(out_corpus)


def extract_raw_input(entry: EnrichedWebNLGEntry) -> List[RDFTripleList]:
    triplelist = RDFTripleList([])
    for tripleset in entry.modifiedtripleset.mtriple:
        triplelist.append(RDFTriple.from_string(tripleset))
    return triplelist


def extract_selected_input_from_lex(lex) -> RDFTripleList:
    triplelist = []
    for sentence in lex.sortedtripleset.sentence:
        for sortedtriple in sentence.striple:
            triplelist.append(RDFTriple.from_string(sortedtriple))
    return RDFTripleList(triplelist)


def extract_ordered_input_from_lex(lex) -> RDFTripleList:
    # WebNLG presents selected triples in their order of realisation.
    return extract_selected_input_from_lex(lex)


def extract_sentence_segmented_input_from_lex(lex) -> Tuple[Tuple[RDFTriple]]:
    selected_inputs = []
    for sentence in lex.sortedtripleset.sentence:
        triplelist = []
        for sortedtriple in sentence.striple:
            triplelist.append(RDFTriple.from_string(sortedtriple))
        selected_inputs.append(tuple(triplelist))
    return tuple(selected_inputs)


def extract_lexicalization(entry: EnrichedWebNLGEntry) -> List[str]:
    return [target.lexicalization for target in entry.lex]


def extract_reg(entry: EnrichedWebNLGEntry) -> List[str]:
    texts = [target.text for target in entry.lex]
    templates = [target.template for target in entry.lex]
    lexes = [target.lexicalization for target in entry.lex]
    reg_lexes = []
    for text, template, lex in zip(texts, templates, lexes):
        reg_lexes.append(extract_reg_from_lex(text, template, lex))
    return reg_lexes


def extract_refs_from_xsdata_rep(lex_references):
    return EnrichedWebNLGReferences([EnrichedWebNLGReference(ref.entity, ref.number, ref.tag, ref.type_value, ref.value) for ref in lex_references])


def extract_reg_from_lex(text, template, lex):
    # TODO rewrite this to use the entry.lex.references and a sem_class_json file
    if None in (text, template, lex):
        print(text)
        print(template)
        print(lex)
        return None
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
                    parts = [curr_rest[i:] for i in range(len(target_tokens))]
                    for start_idx, token_tuple in enumerate(zip(*parts)):
                        # print(token_tuple)
                        if token_tuple == target_tokens:
                            new_lex.extend(target_tokens)
                            curr_text_idx += start_idx + len(target_tokens)
                            match_found = True
                            break
                else:
                    message = "Must have possible targets for each slot!"
                    raise ValueError(message)
                if match_found:
                    break
            if not match_found:
                logger.debug(f"Could not create reg_lex text for {lex_token}")
                logger.debug(f"in:\n{text}\n{template}\n{lex}\n{reg_dict}")
                extract_reg_from_template_and_text(text, template, print_diff=True)
                new_lex.append(lex_token)
                # raise ValueError(f"Could not create reg_lex text for {lex_token} in:\n{text}\n{template}\n{lex}\n{reg_dict}")
        else:
            new_lex.append(lex_token)
    return " ".join(new_lex)


def extract_raw_output(entry: EnrichedWebNLGEntry) -> List[str]:
    return [target.text for target in entry.lex]


def load_enriched_webnlg(enriched_webnlg_config: Optional[omegaconf.DictConfig] = None,
                         splits: Optional[Iterable[str]] = None,
                         undo_enriched_webnlg_delex: bool = True) -> EnrichedWebNLGCorpus:
    """
    :param enriched_webnlg_config: an omegaconf.DictConfig like object containing the basic
                                   information about the e2e corpus to be used
    :param splits: which splits to load
    :return: the corpus of RDF-text pairs with metadata
    """
    default_splits = set(enriched_webnlg_config.splits.keys())
    if not set(splits).issubset(default_splits):
        message = f"`splits` can only contain a subset of {default_splits}. Found {splits}."
        raise ValueError(message)
    data_directory = Path(__file__).parent / enriched_webnlg_config.load_dir
    fns = []
    for split in splits:
        logger.info(split)
        for tuple_dir in os.listdir(os.path.join(data_directory, split)):
            fns.extend([os.path.join(data_directory, split, tuple_dir, fn) for fn in os.listdir(os.path.join(data_directory, split, tuple_dir))])

    corpus: EnrichedWebNLGCorpusRaw = EnrichedWebNLGCorpusRaw(filename_or_list=fns)
    corpus.metadata = {'name': enriched_webnlg_config.display_name,
                       'splits': splits,
                       'directory': data_directory,
                       'raw': True}
    logger.info(len(corpus))

    # tokenize texts
    for entry in corpus:
        for target in entry.lex:
            target.text = TGenTokeniser.tokenise(target.text)

    # Specify the type again since we're changing the expected type of the variable and mypy doesn't like that
    corpus: EnrichedWebNLGCorpus = EnrichedWebNLGCorpus.from_raw_corpus(corpus)
    corpus.metadata = {'name': enriched_webnlg_config.display_name,
                       'splits': splits,
                       'directory': data_directory,
                       'raw': False}
    logger.info(f"Corpus contains {len(corpus)} entries.")

    if undo_enriched_webnlg_delex:
        for entry in corpus:
            entry.undo_enriched_webnlg_delex()
    return corpus


def sanitize_subjects_and_objects(subject_or_object: str) -> List[str]:
    out_string = subject_or_object.replace(",", " , ")
    out_tokens = out_string.split()
    # omit empty strings caused by multiple spaces in a row
    return [token for token in out_tokens if token]


def sanitize_predicates(predicate: str) -> List[str]:
    return [predicate]


snake_case_regex = regex.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')


def tokenize_slots_and_values(value):
    if value.startswith("__") and value.endswith("__"):
        return [value]
    else:
        out_string = snake_case_regex.sub(r'_\1', value)
        out_string = out_string.replace("_", " ").replace(",", " , ")
        out_tokens = out_string.split()
        # omit empty strings caused by multiple spaces in a row
        return [token for token in out_tokens if token]


def linearize_rdf_triple_list(rdf_triple_list: Union[List[RDFTriple], RDFTripleList]) -> List[str]:
    tokens = ["<RDF_TRIPLES>"]
    for rdf_triple in rdf_triple_list:
        tokens.append("<SUBJECT>")
        tokens.extend(tokenize_slots_and_values(rdf_triple.subject))
        tokens.append("</SUBJECT>")
        tokens.append("<PREDICATE>")
        tokens.extend([x.lower() for x in tokenize_slots_and_values(rdf_triple.predicate)])
        tokens.append("</PREDICATE>")
        tokens.append("<OBJECT>")
        tokens.extend(tokenize_slots_and_values(rdf_triple.object))
        tokens.append("</OBJECT>")
        tokens.append("<TRIPLE_SEP>")
    tokens.append("</RDF_TRIPLES>")
    return tokens


def linearize_seq_of_rdf_triple_lists(seq_of_rdf_triple_lists) -> List[str]:
    tokens = []
    for rdf_triple_list in seq_of_rdf_triple_lists:
        tokens.append("<SENTENCE>")
        tokens.extend(linearize_rdf_triple_list(rdf_triple_list))
        tokens.append("</SENTENCE>")
    return tokens

LINEARIZATION_FUNCTIONS = {'raw_input': linearize_rdf_triple_list,
                           'selected_input': linearize_rdf_triple_list,
                           'ordered_input': linearize_rdf_triple_list,
                           'sentence_segmented_input': linearize_seq_of_rdf_triple_lists,
                           'lexicalisation': lambda lex_string: lex_string.strip().split(),
                           'referring_expressions': lambda reg_string: reg_string.strip().split(),
                           'raw_output': lambda text: text.strip().split()}
