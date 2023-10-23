import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

import enunlg.data_management.enriched_e2e as ee2e
import enunlg.encdec.seq2seq as s2s

class PipelineSeq2SeqGenerator(object):
    def __init__(self, corpus: ee2e.EnrichedE2ECorpus):
        self.layers = corpus.layers
        self.pipeline = corpus.views
        self.modules = {layer_pair: None for layer_pair in self.pipeline}
        self.vocabularies = {layer: None for layer in self.layers}
        self.initialize_vocabularies(corpus)
        self.initialize_seq2seq_modules(corpus)

    def initialize_vocabularies(self, corpus):
        pass

    def initialize_seq2seq_modules(self, corpus):

        pass

    def predict(self, mr):
        curr_input = mr
        for layer_pair in self.modules:
            logging.debug(layer_pair)
            logging.debug(curr_input)
            curr_output = self.modules[layer_pair].predict(curr_input)
            logging.debug(curr_output)
            curr_input = curr_output
        return curr_output


if __name__ == "__main__":
    corpus = ee2e.load_enriched_e2e(splits=("dev", ))
    for x in corpus[:6]:
        print(x)

    plg = PipelineLookupGenerator(corpus)
    for entry in corpus:
        mr = entry.raw_input
        print(mr)
        print(plg.predict(mr))
        print("----")
