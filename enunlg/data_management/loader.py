from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Tuple

import json
import logging

import omegaconf

import enunlg.data_management.cued
import enunlg.data_management.e2e_challenge
import enunlg.data_management.enriched_e2e
import enunlg.data_management.enriched_webnlg
import enunlg.util


if TYPE_CHECKING:
    from enunlg.data_management.pipelinecorpus import AnyPipelineCorpus, TextPipelineCorpus


logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"e2e", "e2e-cleaned", "e2e-enriched", "webnlg-enriched", "sfx-restaurant"}
SUPPORTED_PIPELINE_DATASETS = {"e2e-enriched", "webnlg-enriched"}


def load_data_from_config(data_config: "omegaconf.DictConfig", splits, sem_class_delex: Optional[str] = None):
    """Selects the right function to use to load the desired data and return a corpus.

    In an experiment's YAML file, the data config is specified under the key `data`.
    Expected properties are `corpus`, which has a name (string) and a list of (named) splits (as strings).
    The `data` node also has an `input mode`, specifying whether to use the `e2e` input mode (i.e. SlotValueMRs)
    or `rdf` input mode.

    Example
    -------
    data:
      corpus:
        name: enriched-e2e
        splits: [train]
      input_mode: e2e
    """
    if data_config.corpus.name not in SUPPORTED_DATASETS:
        message = f"Unsupported dataset: {data_config.corpus.name}"
        raise ValueError(message)
    if data_config.corpus.name == 'e2e':
        logger.info(f"Loading E2E Challenge Data ({splits})...")
        return enunlg.data_management.e2e_challenge.load_e2e(data_config.corpus, splits)
    elif data_config.corpus.name == 'e2e-cleaned':
        logger.info(f"Loading the Cleaned E2E Data ({splits})...")
        return enunlg.data_management.e2e_challenge.load_e2e(data_config.corpus, splits)
    elif data_config.corpus.name == 'e2e-enriched':
        logger.info("Loading Enriched E2E Challenge Data...")
        return enunlg.data_management.enriched_e2e.load_enriched_e2e(data_config.corpus, splits)
    elif data_config.corpus.name == 'webnlg-enriched':
        logger.info("Loading Enriched WebNLG (v1.6) Data...")
        return enunlg.data_management.enriched_webnlg.load_enriched_webnlg(data_config.corpus, splits, undo_enriched_webnlg_delex=False)
    elif data_config.corpus.name == 'sfx-restaurant':
        logger.info("Loading SFX Restaurant data...")
        return enunlg.data_management.cued.load_sfx_restaurant(data_config.corpus.splits)
    else:
        message = f"It should not be possible to get this error message. You tried to use {data_config.corpus.name}"
        raise ValueError(message)


def prep_pipeline_corpus(config: omegaconf.DictConfig,
                         splits: List[str],
                         delexicalise: bool = True,
                         print_summaries: bool = True) -> "Tuple[AnyPipelineCorpus, AnyPipelineCorpus, TextPipelineCorpus]":
    pipeline_corpus = load_data_from_config(config, splits)
    if print_summaries:
        pipeline_corpus.print_summary_stats()

    if config.corpus.name == "e2e-enriched":
        pipeline_corpus.validate_enriched_e2e()
        if delexicalise:
            pipeline_corpus.delexicalise_by_slot_name(('name', 'near'))
        slot_value_corpus = deepcopy(pipeline_corpus)
        if config.input_mode == "rdf":
            enunlg.util.translate_sv_corpus_to_rdf(pipeline_corpus)
    elif config.corpus.name == "webnlg-enriched":
        if delexicalise:
            sem_class_dict = json.load(Path("datasets/processed/enriched-webnlg.dbo-delex.70-percent-coverage.json").open('r'))
            sem_class_lower = {key.lower(): sem_class_dict[key] for key in sem_class_dict}
            pipeline_corpus.delexicalise_with_sem_classes(sem_class_lower)
        slot_value_corpus = deepcopy(pipeline_corpus)
        # For some reason metadata doesn't get copied???
        slot_value_corpus.metadata = pipeline_corpus.metadata
        # It's not yet a slot-value corpus until we run this
        enunlg.util.translate_rdf_corpus_to_e2e(slot_value_corpus)
        if config.input_mode == "e2e":
            pipeline_corpus = slot_value_corpus
    else:
        raise ValueError(f"Prepare pipeline corpus can only work with {SUPPORTED_PIPELINE_DATASETS}")

    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    if config.input_mode == "rdf":
        lin_func_label = "enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS"
        linearisation_functions = enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS
    elif config.input_mode == "e2e":
        lin_func_label = "enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS"
        linearisation_functions = enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS
        if config.corpus.name == "webnlg-enriched":
            lin_func_label = "enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS_WITH_SLOTVALUE_LISTS"
            linearisation_functions = enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS_WITH_SLOTVALUE_LISTS
    text_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus.from_existing(pipeline_corpus, mapping_functions=linearisation_functions)
    text_corpus.metadata['linearisation_functions'] = lin_func_label
    if print_summaries:
        text_corpus.print_summary_stats()
        text_corpus.print_sample(0, 100, 10)
    return pipeline_corpus, slot_value_corpus, text_corpus
