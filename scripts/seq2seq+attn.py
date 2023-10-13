import logging
import random

import omegaconf
import hydra
import torch

import enunlg.encdec.seq2seq as s2s
import enunlg.trainer
import enunlg.vocabulary


class NNGenerator(object):
    def __init__(self, input_vocab, output_vocab, model_class, model_config):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.model = model_class(model_config)

    def generate_from_input_seq(self, input_seq):
        return self._output_bridge(self.model.generate(self._input_bridge(input_seq)))
        pass

    def _input_bridge(self, input_seq):
        """convert input into input appropriate for self.model"""

    def _output_bridge(self, output_seq):
        """Convert raw output of self.model into output"""
        pass


@hydra.main(version_base=None, config_path='../config', config_name='seq2seq+attn')
def seq2seq_attn_main(config: omegaconf.DictConfig):
    hydra_managed_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    seed = config.random_seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Prepare mock data
    lowercase = "abcdefghijklmnopqrstuvwxyz"
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    items = []
    for _ in range(1000):
        lb = random.choice(range(26))
        ub = random.choice(range(lb, 26))
        items.append((" ".join(lowercase[lb:ub]).split(),
                      " ".join(uppercase[lb:ub]).split()))
    train, dev, test = items[:800], items[800:900], items[900:]

    print(len(train))

    input_vocab = enunlg.vocabulary.TokenVocabulary(lowercase)
    output_vocab = enunlg.vocabulary.TokenVocabulary(uppercase)

    def prep_embeddings(vocab1, vocab2, tokens):
        return [(torch.tensor(vocab1.get_ints_with_left_padding(x[0], 26-2), dtype=torch.long),
                 torch.tensor(vocab2.get_ints(x[1]), dtype=torch.long)) for x in tokens]
    train_embeddings = prep_embeddings(input_vocab, output_vocab, train)
    dev_embeddings = prep_embeddings(input_vocab, output_vocab, dev)

    model = s2s.Seq2SeqAttn(input_vocab.size, output_vocab.size, model_config=config.model)
    print(model)

    trainer = enunlg.trainer.Seq2SeqAttnTrainer(model, training_config=config.mode.train, input_vocab=input_vocab, output_vocab=output_vocab)

    trainer.train_iterations(train_embeddings, dev_embeddings)


if __name__ == "__main__":
    seq2seq_attn_main()
