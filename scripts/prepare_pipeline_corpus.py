from pathlib import Path

import json
import logging
import sys

import omegaconf
import hydra

from enunlg.data_management.loader import load_data_from_config

import enunlg.data_management.enriched_e2e
import enunlg.data_management.enriched_webnlg
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.multitask_seq2seq
import enunlg.trainer.multitask_seq2seq
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger('enunlg-scripts.prepare_pipeline_corpus')

SUPPORTED_DATASETS = {"enriched-e2e", "enriched-webnlg"}


def sem_class_dict_from_mille(json_filepath, sem_class_type: str = 'class_dbp'):
    with Path(json_filepath).open('r') as json_file:
        sem_class_data = json.load(json_file)
    sem_class_lower = {}
    for k in sem_class_data:
        curr_val = sem_class_data[k][sem_class_type]
        if curr_val not in ("", "—"):
            sem_class_lower[k.lower()] = curr_val
    return sem_class_lower


def prep_corpus(config: omegaconf.DictConfig) -> enunlg.data_management.pipelinecorpus.TextPipelineCorpus:
    pipeline_corpus = load_data_from_config(config, splits=["dev"])

    if config.corpus.name == "e2e-enriched":
        enunlg.data_management.enriched_e2e.validate_enriched_e2e(pipeline_corpus)

    sem_class_dict = json.load(Path("datasets/processed/enriched-webnlg.dbo-delex.70-percent-coverage.json").open('r'))
    sem_class_lower = {key.lower(): sem_class_dict[key] for key in sem_class_dict}
    pipeline_corpus.delexicalise_with_sem_classes(sem_class_lower)


    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    if config.input_mode == "rdf":
        linearization_functions = enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS
    elif config.input_mode == "e2e":
        linearization_functions = enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS
    return enunlg.data_management.pipelinecorpus.TextPipelineCorpus.from_existing(pipeline_corpus, mapping_functions=linearization_functions)


@hydra.main(version_base=None, config_path='../config/data', config_name='enriched-webnlg_as-rdf')
def prep_pipeline_corpus_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    corpus = prep_corpus(config)

    corpus.write_to_iostream(Path("webnlg.delex.tmp").open("w"))
    # corpus.write_to_iostream(sys.stdout)


if __name__ == "__main__":
    prep_pipeline_corpus_main()