import logging
import random

import omegaconf
import hydra
import torch

import enunlg.encdec.seq2seq as s2s
import enunlg.trainer
import enunlg.vocabulary


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

    model = s2s.TGenEncDec(input_vocab, output_vocab, model_config=config.model)
    print(model)

    trainer = enunlg.trainer.TGenTrainer(model, training_config=config.mode.train)

    trainer.train_iterations(train_embeddings, dev_embeddings)


if __name__ == "__main__":
    seq2seq_attn_main()
