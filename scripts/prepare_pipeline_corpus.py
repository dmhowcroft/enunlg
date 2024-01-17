import logging
import sys

import enunlg.data_management.enriched_e2e
import enunlg.data_management.enriched_webnlg
import enunlg.data_management.pipelinecorpus as plc

logger = logging.getLogger('enunlg-scripts.prepare_pipeline_corpus')

SUPPORTED_DATASETS = {"enriched-e2e", "enriched-webnlg"}



def load_data_from_config(data_config: "omegaconf.DictConfig"):
    if data_config.corpus.name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {data_config.corpus.name}")
    if data_config.corpus.name == 'enriched-e2e':
        logger.info("Loading Enriched E2E Challenge Data...")
        return enunlg.data_management.enriched_e2e.load_enriched_e2e(data_config.corpus.splits)
    elif data_config.corpus.name == 'enriched-webnlg':
        logger.info("Loading Enriched WebNLG (v1.6) Data...")
        return enunlg.data_management.enriched_webnlg.load_enriched_webnlg(data_config.corpus.splits)
    else:
        raise ValueError("We can only load the Enriched E2E and Enriched WebNLG datasets right now.")



if __name__ == "__main__":
    import omegaconf
    tmp_config = omegaconf.DictConfig({"corpus": {"name": "enriched-e2e",
                                                  "splits": ["train"]}})
    # tmp_config = omegaconf.DictConfig({"corpus": {"name": "enriched-webnlg",
    #                                               "splits": ["train"]}})

    pipeline_corpus = load_data_from_config(tmp_config)
    pipeline_corpus.print_summary_stats()
    print("____________")

    # Convert annotations from datastructures to 'text' -- i.e. linear sequences of a specific type.
    if tmp_config.corpus.name == "enriched-webnlg":
        linearization_functions = enunlg.data_management.enriched_webnlg.LINEARIZATION_FUNCTIONS
    elif tmp_config.corpus.name == "enriched-e2e":
        linearization_functions = enunlg.data_management.enriched_e2e.LINEARIZATION_FUNCTIONS
    text_corpus = plc.TextPipelineCorpus.from_existing(pipeline_corpus, mapping_functions=linearization_functions)
    text_corpus.print_summary_stats()

    text_corpus.write_to_iostream(sys.stdout)

