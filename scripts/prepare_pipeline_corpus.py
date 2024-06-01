from pathlib import Path

import logging
import sys

import omegaconf
import hydra

from enunlg.data_management.loader import prep_pipeline_corpus
from enunlg.meaning_representation.slot_value import SlotValueMRList

logger = logging.getLogger('enunlg-scripts.prepare_pipeline_corpus')


@hydra.main(version_base=None, config_path='../config/data', config_name='enriched-webnlg_as-rdf_rdf-roles-delex')
def prep_pipeline_corpus_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    corpus, sv_corpus, text_corpus = prep_pipeline_corpus(config, ['dev'])
    # text_corpus.drop_layers(keep=["raw_input"])

    # text_corpus.write_to_iostream(Path("enriched-e2e_rdf.txt").open("w"))
    text_corpus.write_to_iostream(sys.stdout)

    unique_entities = set()
    unique_predicates = set()
    unique_predicate_object_pairs = set()
    unique_objects = set()
    for entry in sv_corpus:

        mr_list = [entry['raw_input']]
        # print(mr_list)
        for mr in mr_list:
            if isinstance(mr, SlotValueMRList):
                retval = []
                for sub_mr in mr:
                    if 'name' in sub_mr:
                        retval.append(sub_mr['name'])
                if len(retval) == 1:
                    [unpacked] = retval
                    unique_entities.add(unpacked)
                elif len(retval) == 0:
                    unique_entities.add(None)
                else:
                    for entity in retval:
                        unique_entities.add(entity)
                for sub_mr in mr:
                    for slot in sub_mr:
                        if slot != "name":
                            unique_predicates.add(slot)
                            unique_predicate_object_pairs.add((slot, sub_mr[slot]))
                            unique_objects.add(sub_mr[slot])
            else:
                unique_entities.add(mr.get('name'))
                for slot in mr:
                    if slot != "name":
                        unique_predicates.add(slot)
                        unique_predicate_object_pairs.add((slot, mr[slot]))
                        unique_objects.add(mr[slot])

    print(unique_entities)
    print(len(unique_entities))
    print(len(unique_predicates))
    print(len(unique_predicate_object_pairs))
    print(len(unique_objects))


if __name__ == "__main__":
    prep_pipeline_corpus_main()