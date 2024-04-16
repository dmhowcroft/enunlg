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


def delexicalise_with_sem_classes(pipeline_corpus: "enunlg.data_management.enriched_webnlg.EnrichedWebNLGCorpus",
                                  sem_class_json) -> "enunlg.data_management.enriched_webnlg.EnrichedWebNLGCorpus":
    with Path(sem_class_json).open('r') as json_file:
        sem_class_data = json.load(json_file)

    present = 0
    absent = 0
    for entry in pipeline_corpus:
        # check if entities are in sem_class_data
        print(entry.reg_dict)
        for delex_key in entry.reg_dict:
            values = entry.reg_dict[delex_key]
            # loop over values until we find one in sem_class_data
            # if we found one, create a new dict entry mapping the old class to the new one
            # incorporate tehse changes into extract_reg_from_lex so so we can call the new
            #   method in raw_to_usable to get what we need
    # print(present / (present+absent))
    return pipeline_corpus



def prep_corpus(tmp_config) -> None:
    pipeline_corpus = load_data_from_config(tmp_config, splits=["dev"])

    if tmp_config.corpus.name == "e2e-enriched":
        enunlg.data_management.enriched_e2e.validate_enriched_e2e(pipeline_corpus)

    json_filepath = Path("datasets/raw/2024-04-12_mille_webnlg_dbp-wkd-classes.json")
    pipeline_corpus = delexicalise_with_sem_classes(pipeline_corpus, json_filepath)

    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    if tmp_config.input_mode == "rdf":
        linearization_functions = enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS
    elif tmp_config.input_mode == "e2e":
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