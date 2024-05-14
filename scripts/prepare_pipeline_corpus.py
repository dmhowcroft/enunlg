from pathlib import Path
from typing import Dict

import json
import logging

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

def delexicalise_with_sem_classes(pipeline_corpus: "enunlg.data_management.enriched_webnlg.EnrichedWebNLGCorpus",
                                  sem_class_dict: Dict[str, str]) -> "enunlg.data_management.enriched_webnlg.EnrichedWebNLGCorpus":
    present = 0
    absent = 0
    for entry in pipeline_corpus:
        # check if entities are in sem_class_data
        logger.debug("-=-=-=-=-=-=-=-==-")
        logger.debug(entry)
        for reference in entry.references.sequence:
            entity = reference.entity
            logger.debug(f"===> entity: {entity}")
            logger.debug(f"---> original tag: {entry.references.entity_orig_tag_mapping[entity]}")
            if entity.lower() in sem_class_dict:
                dbpedia_class = sem_class_dict[entity.lower()]
                entry.delex_reference(entity, dbpedia_class)
                present +=1
            else:
                absent += 1
            # if we found one, create a new dict entry mapping the old class to the new one
            # incorporate these changes into extract_reg_from_lex so so we can call the new
            #   method in raw_to_usable to get what we need
    logger.info(f"Percentage of entities for which we have an entry in the sem_class_dict: {present / (present + absent)}")
    return pipeline_corpus


def prep_corpus(config: omegaconf.DictConfig) -> enunlg.data_management.pipelinecorpus.TextPipelineCorpus:
    pipeline_corpus = load_data_from_config(config, splits=["dev"])

    if config.corpus.name == "e2e-enriched":
        enunlg.data_management.enriched_e2e.validate_enriched_e2e(pipeline_corpus)

    sem_class_dict = sem_class_dict_from_mille("datasets/raw/2024-04-12_mille_webnlg_dbp-wkd-classes.json")
    pipeline_corpus = delexicalise_with_sem_classes(pipeline_corpus, sem_class_dict)

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


    # corpus.write_to_iostream(sys.stdout)


if __name__ == "__main__":
    prep_pipeline_corpus_main()