from typing import TYPE_CHECKING

import logging

import enunlg.data_management.cued
import enunlg.data_management.e2e_challenge
import enunlg.data_management.enriched_e2e
import enunlg.data_management.enriched_webnlg

if TYPE_CHECKING:
    import omegaconf

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"e2e", "e2e-cleaned", "e2e-enriched", "enriched-webnlg", "sfx-restaurant"}


def load_data_from_config(data_config: "omegaconf.DictConfig", splits):
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
        raise ValueError(f"Unsupported dataset: {data_config.corpus.name}")
    if data_config.corpus.name == 'e2e':
        logger.info(f"Loading E2E Challenge Data ({splits})...")
        return enunlg.data_management.e2e_challenge.load_e2e(data_config.corpus, splits)
    elif data_config.corpus.name == 'e2e-cleaned':
        logger.info(f"Loading the Cleaned E2E Data ({splits})...")
        return enunlg.data_management.e2e_challenge.load_e2e(data_config.corpus, splits)
    elif data_config.corpus.name == 'e2e-enriched':
        logger.info("Loading Enriched E2E Challenge Data...")
        return enunlg.data_management.enriched_e2e.load_enriched_e2e(data_config.corpus, splits)
    elif data_config.corpus.name == 'enriched-webnlg':
        logger.info("Loading Enriched WebNLG (v1.6) Data...")
        return enunlg.data_management.enriched_webnlg.load_enriched_webnlg(data_config.corpus.splits)
    elif data_config.corpus.name == 'sfx-restaurant':
        logger.info("Loading SFX Restaurant data...")
        return enunlg.data_management.cued.load_sfx_restaurant(data_config.corpus.splits)
    else:
        raise ValueError("We can only load the Enriched E2E and Enriched WebNLG datasets right now.")
