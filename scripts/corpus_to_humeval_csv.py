from pathlib import Path

import logging
import random

import pandas as pd

from enunlg.data_management.pipelinecorpus import TextPipelineCorpus


logger = logging.getLogger('enunlg-scripts.corpus_to_humeval_csv')


if __name__ == "__main__":
    corpus_fp = Path("for-analysis/llm/webnlg_rdf.txt")
    text_corpus = TextPipelineCorpus.load(corpus_fp)
    ids = []
    for idx in range(1, 500, 10):
        i2 = random.choice((1, 2, 3))
        print(f"Id{idx}-Id{i2}")
        ids.append(f"Id{idx}-Id{i2}")
    annotation_layers = ["raw_input", "GPT4_output", "Llama_output"]
    rows = []
    text_corpus.print_sample()
    for entry in text_corpus:
        # print(entry)
        if entry.metadata.get('id') in ids:
            row = [entry.metadata.get('id')]
            for layer_name in annotation_layers:
                row.append(entry[layer_name])
            rows.append(row)
    df = pd.DataFrame(rows, columns=["id"] + annotation_layers)
    print(df)
