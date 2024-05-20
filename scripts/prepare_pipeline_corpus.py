import logging
import sys

import omegaconf
import hydra

from enunlg.data_management.loader import prep_pipeline_corpus

logger = logging.getLogger('enunlg-scripts.prepare_pipeline_corpus')


@hydra.main(version_base=None, config_path='../config/data', config_name='enriched-webnlg_as-e2e')
def prep_pipeline_corpus_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    corpus, sv_corpus, text_corpus = prep_pipeline_corpus(config, ['train'])

    # text_corpus.write_to_iostream(Path("webnlg.delex.tmp").open("w"))
    # text_corpus.write_to_iostream(sys.stdout)


if __name__ == "__main__":
    prep_pipeline_corpus_main()