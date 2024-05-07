from collections import defaultdict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import difflib
import json
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


class EnrichedWebNLGItem(enunlg.data_management.pipelinecorpus.PipelineItem):
    def __init__(self, annotation_layers):
        super().__init__(annotation_layers)
        self.reg_dict = {}


class EnrichedWebNLGCorpus(enunlg.data_management.pipelinecorpus.PipelineCorpus):
    def __init__(self, seq: List[EnrichedWebNLGItem], metadata=None):
        super(EnrichedWebNLGCorpus, self).__init__(seq, metadata)


def extract_raw_input(entry: EnrichedWebNLGEntry) -> List[RDFTripleList]:
    triplelist = RDFTripleList([])
    for tripleset in entry.modifiedtripleset.mtriple:
        triplelist.append(RDFTriple.from_string(tripleset))
    return triplelist


def extract_selected_input(entry: EnrichedWebNLGEntry) -> List[RDFTripleList]:
    return [extract_selected_input_from_lex(lex) for lex in entry.lex]


def extract_selected_input_from_lex(lex) -> RDFTripleList:
    triplelist = []
    for sentence in lex.sortedtripleset.sentence:
        for sortedtriple in sentence.striple:
            triplelist.append(RDFTriple.from_string(sortedtriple))
    return RDFTripleList(triplelist)


def extract_ordered_input(entry: EnrichedWebNLGEntry) -> List[RDFTripleList]:
    targets = []
    for lex in entry.lex:
        triplelist = []
        for sentence in lex.sortedtripleset.sentence:
            for sortedtriple in sentence.striple:
                triplelist.append(RDFTriple.from_string(sortedtriple))
        targets.append(RDFTripleList(triplelist))
    return targets


def extract_ordered_input_from_lex(lex) -> RDFTripleList:
    triplelist = []
    for sentence in lex.sortedtripleset.sentence:
        for sortedtriple in sentence.striple:
            triplelist.append(RDFTriple.from_string(sortedtriple))
    return RDFTripleList(triplelist)


def extract_sentence_segmented_input(entry: EnrichedWebNLGEntry) -> List[Tuple[Tuple[RDFTriple]]]:
    targets = []
    for lex in entry.lex:
        selected_inputs = []
        for sentence in lex.sortedtripleset.sentence:
            triplelist = [RDFTriple.from_string(sortedtriple) for sortedtriple in sentence.striple]
            selected_inputs.append(tuple(triplelist))
        targets.append(tuple(selected_inputs))
    return targets


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


class WebNLGReference(object):
    def __init__(self, entity, seq_loc, orig_delex_tag, ref_type, form):
        self.entity = str(entity)
        self.seq_loc = seq_loc
        self.orig_delex_tag = orig_delex_tag
        self.ref_type = ref_type
        self.form = form

class WebNLGReferences(object):
    def __init__(self, ref_list):
        self.sequence = ref_list
        self.lookup_by_entity = defaultdict(list)
        for index, ref in enumerate(self.sequence):
            self.lookup_by_entity[ref.entity].append(index)


def extract_refs_from_xsdata_rep(lex_references):
    return WebNLGReferences([WebNLGReference(ref.entity, ref.number, ref.tag, ref.type_value, ref.value) for ref in lex_references])


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


def raw_to_usable(raw_corpus) -> List[EnrichedWebNLGItem]:
    """This will drop any entries which contain 'None' for any annotation layers"""
    out_corpus = []
    for entry in raw_corpus:
        raw_input = extract_raw_input(entry)
        for lex in entry.lex:
            selected_input = extract_selected_input_from_lex(lex)
            ordered_input = extract_ordered_input_from_lex(lex)
            sentence_segmented_input = extract_sentence_segmented_input_from_lex(lex)
            lexicalization = lex.lexicalization
            raw_output = lex.text
            if None in [raw_input, selected_input, ordered_input, sentence_segmented_input, raw_output, lex.template, lexicalization]:
                continue
            reg_string = extract_reg_from_lex(raw_output, lex.template, lexicalization)
            if reg_string is None:
                continue
            new_item = EnrichedWebNLGItem({'raw_input': raw_input,
                                                  'selected_input': selected_input,
                                                  'ordered_input': ordered_input,
                                                  'sentence_segmented_input': sentence_segmented_input,
                                                  'lexicalisation': lexicalization,
                                                  'referring_expressions': reg_string,
                                                  'raw_output': raw_output})
            new_item.reg_dict = extract_reg_from_template_and_text(raw_output, lex.template)
            new_item.references = extract_refs_from_xsdata_rep(lex.references.reference)
            out_corpus.append(new_item)
    return out_corpus


def load_enriched_webnlg(enriched_webnlg_config: Optional[omegaconf.DictConfig] = None,
                         splits: Optional[Iterable[str]] = None,
                         sem_class_delex: Optional[str] = None) -> EnrichedWebNLGCorpus:
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
    corpus: EnrichedWebNLGCorpus = EnrichedWebNLGCorpus(raw_to_usable(corpus))
    corpus.metadata = {'name': enriched_webnlg_config.display_name,
                       'splits': splits,
                       'directory': data_directory,
                       'raw': False}
    logger.info(f"Corpus contains {len(corpus)} entries.")
    return corpus


def sanitize_subjects_and_objects(subject_or_object: str) -> List[str]:
    out_string = subject_or_object.replace("_", " ").replace(",", " , ")
    out_tokens = out_string.split()
    # omit empty strings caused by multiple spaces in a row
    return [token for token in out_tokens if token]


def sanitize_predicates(predicate: str) -> List[str]:
    return [predicate]




snake_case_regex = regex.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')


def tokenize_slots_and_values(value):
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
