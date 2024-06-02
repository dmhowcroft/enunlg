from pathlib import Path
from typing import List

import logging
import random

import pandas as pd

from enunlg.data_management.pipelinecorpus import TextPipelineCorpus


logger = logging.getLogger('enunlg-scripts.corpus_to_humeval_csv')

corpus_files = {
    'multitask_webnlg_sv': None,
    'multitask_webnlg_rdf': None,
    'multitask_webnlg_rdf_role-delex': None,
    'multitask_webnlg_rdf_agent-pred-delex': None,
    'singletask_webnlg_sv': None,
    'singletask_webnlg_rdf': None,
    'singletask_webnlg_rdf_role-delex': None,
    'singletask_webnlg_rdf_agent-pred-delex': None,
    'llm_webnlg_sv': "for-analysis/llm/webnlg_slot-value.txt",
    'llm_webnlg_rdf': "for-analysis/llm/webnlg_rdf.txt",
    'multitask_e2e_sv': None,
    'multitask_e2e_rdf': None,
    'multitask_e2e_rdf_role-delex': None,
    'multitask_e2e_rdf_agent-pred-delex': None,
    'singletask_e2e_sv': None,
    'singletask_e2e_rdf': None,
    'singletask_e2e_rdf_role-delex': None,
    'singletask_e2e_rdf_agent-pred-delex': None,
    'llm_e2e_sv': "for-analysis/llm/e2e_slot-value.txt",
    'llm_e2e_rdf': "for-analysis/llm/e2e_rdf.txt",
}


def sample_ids() -> List[str]:
    sample = []
    for idx in range(1, 500, 10):
        i2 = random.choice((1, 2, 3))
        print(f"Id{idx}-Id{i2}")
        sample.append(f"Id{idx}-Id{i2}")
    return sample


if __name__ == "__main__":
    # Load all the results corpora
    corpora_for_analysis = {}
    for sys_corpus_format_delex in corpus_files:
        if corpus_files[sys_corpus_format_delex] is not None:
            corpus_fp = Path(corpus_files[sys_corpus_format_delex])
            corpora_for_analysis[sys_corpus_format_delex] = TextPipelineCorpus.load(corpus_fp)

    ids = sample_ids()

    metadata_columns = ["id", "system", "corpus", "format", "delex"]
    dfs_for_analysis = {}
    for key in corpora_for_analysis:
        print(key)
        parts = key.split("_")
        system = parts[0]
        corpus = parts[1]
        mr_type = parts[2]
        if parts[3:]:
            delex = parts[3]
        else:
            if corpus == "e2e":
                delex = 'name-near_exact-match'
            else:
                delex = 'dbpedia-ontology-classes'
        df_metadata = [system, corpus, mr_type, delex]
        print(df_metadata)
        if "llm" in key:
            annotation_layers = ["raw_input", "GPT4_output", "Llama_output"]
        else:
            annotation_layers = ["raw_input", "best_output_relexed"]
        rows = []
        for entry in corpora_for_analysis[key]:
            if entry.metadata.get('id') in ids:
                row = [entry.metadata.get('id')] + df_metadata
                for layer_name in annotation_layers:
                    row.append(entry[layer_name])
                rows.append(row)
        dfs_for_analysis[key] = pd.DataFrame(rows, columns=metadata_columns + annotation_layers)
    print(len(dfs_for_analysis))
