from typing import List, Optional, Tuple

import logging

import omegaconf
import torch
import torch.nn
import torch.nn.functional

import enunlg.encdec.seq2seq as s2s

DEVICE = torch.device("cpu")


class MultitaskLSTMEncoder(s2s.BasicLSTMEncoder):
    def __init__(self, num_unique_inputs, num_embedding_dims, num_hidden_dims, task_names, init="zeros"):
        """
        :param num_unique_inputs:
        :param num_embedding_dims:
        :param num_hidden_dims:
        :param task_names:
        :param init:
        """
        super(MultitaskLSTMEncoder, self).__init__(num_unique_inputs, num_embedding_dims, num_hidden_dims, init)
        self.task_names = task_names
        self.num_tasks = len(task_names)

        # First layer is self.lstm
        self.layers = {self.task_names[0]: self.lstm}
        for task_name in task_names[1:]:
            self.layers[task_name] = torch.nn.LSTM(self.num_hidden_dims, self.num_hidden_dims, batch_first=True)

        if init == "zeros":
            self._hidden_state_init_func = torch.zeros
        else:
            self._hidden_state_init_func = lambda *args: torch.randn(*args)/torch.sqrt(torch.Tensor([self.num_hidden_dims]))

    def forward(self, input_indices, h_c_state):
        # This assumes batch first and batch size = 1
        embedded = self.embedding(input_indices).view(1, len(input_indices), -1)
        outputs, h_c_state = [], []
        layer_outputs, layer_h_c_state = self.lstm(embedded, h_c_state)
        outputs.append(layer_outputs)
        h_c_state.append(layer_h_c_state)
        for task in self.task_names[1:]:
            layer_outputs, layer_h_c_state = self.layers[task](outputs[-1], h_c_state[-1])
            outputs.append(layer_outputs)
            h_c_state.append(layer_h_c_state)
        return outputs, h_c_state

    def forward_one_step(self, input_index, h_c_state):
        raise NotImplementedError()

    def initial_h_c_state(self):
        # This assumes batch first and batch size = 1
        return (self._hidden_state_init_func(1, 1, self.num_hidden_dims, device=DEVICE),
                self._hidden_state_init_func(1, 1, self.num_hidden_dims, device=DEVICE))


class MultiDecoderSeq2SeqAttn(torch.nn.Module):
    def __init__(self, layer_names, layer_vocab_sizes, model_config):
        """
        :param input_vocab:
        :param output_vocab:
        :param model_config:
        """
        super().__init__()
        self.config = model_config

        # Set basic properties
        self.layer_names = layer_names
        self.layer_vocab_sizes = layer_vocab_sizes

        # Initialize encoder and decoder networks
        self.encoder = MultitaskLSTMEncoder(self.input_vocab_size,
                                            self.config.encoder.embeddings.embedding_dim,
                                            self.config.encoder.num_hidden_dims,
                                            self.layer_names[1:])
        self.task_decoders = {}
        for idx, name in enumerate(layer_names[1:]):
            self.task_decoders[name] = s2s.LSTMDecWithAttention(self.config[f"decoder_{name}"].num_hidden_dims,
                                                                self.layer_vocab_sizes[idx+1],
                                                                self.config.max_input_length,
                                                                padding_idx=self.config[f"decoder_{name}"].embeddings.get('padding_idx'),
                                                                start_token_idx=self.config[f"decoder_{name}"].embeddings.get('start_idx'),
                                                                stop_token_idx=self.config[f"decoder_{name}"].embeddings.get('stop_idx'))

    def encode(self, enc_emb: torch.Tensor):
        """
        Run encoding for a single set of integer inputs.

        :param enc_emb: tensor of integer inputs (shape?)
        :return: enc_outputs, enc_h_c_state (final state of the encoder)
        """
        enc_h_c_state = self.encoder.initial_h_c_state()
        # enc_outputs, enc_h_c_state
        return self.encoder(enc_emb, enc_h_c_state)

    def forward(self, enc_emb: torch.Tensor, max_output_length: int = 50):
        enc_outputs, enc_h_c_state = self.encode(enc_emb)

        dec_h_c_state = enc_h_c_state
        dec_input = torch.tensor([[self.decoder.start_idx]], device=DEVICE)
        dec_outputs = [self.decoder.start_idx]

        # run multiple decoders, one for each task
        for dec_index in range(max_output_length):
            dec_output, dec_h_c_state = self.decoder(dec_input, dec_h_c_state, enc_outputs)
            topv, topi = dec_output.data.topk(1)
            dec_outputs.append(topi.item())
            if topi.item() == self.decoder.stop_idx:
                break
            dec_input = topi.squeeze().detach()
        return dec_outputs

    def forward_with_teacher_forcing(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor) -> torch.Tensor:
        enc_outputs, enc_h_c_state = self.encode(enc_emb)

        dec_h_c_state = enc_h_c_state
        # Use torch.zeros because we use padding_idx = 0
        dec_outputs = torch.zeros((len(dec_emb), self.output_vocab_size))
        # the first element of dec_emb is the start token
        dec_outputs[0] = dec_emb[0]
        # That's also why we skip the first element in our loop
        for dec_input_index, dec_input in enumerate(dec_emb[:-1]):
            dec_output, dec_h_c_state = self.decoder(dec_input, dec_h_c_state, enc_outputs)
            dec_outputs[dec_input_index+1] = dec_output
        return dec_outputs

    def train_step(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor, optimizer, criterion):
        optimizer.zero_grad()

        dec_outputs = self.forward_with_teacher_forcing(enc_emb, dec_emb)

        # We should be able to vectorise the following
        dec_targets = torch.tensor([x.unsqueeze(0) for x in dec_emb])
        loss = criterion(dec_outputs, dec_targets)

        loss.backward()
        optimizer.step()
        # mean loss per word returned in order for losses for sents of diff lengths to be comparable
        return loss.item() / dec_emb.size(0)

    def generate(self, enc_emb, max_length=50):
        return self.generate_greedy(enc_emb, max_length)

    def generate_beam(self, enc_emb, max_length=50, beam_size=10, num_expansions: Optional[int] = None):
        if num_expansions is None:
            num_expansions = beam_size
        with torch.no_grad():
            enc_outputs, enc_h_c_state = self.encode(enc_emb)

            dec_h_c_state: torch.Tensor = enc_h_c_state
            prev_beam: List[Tuple[float, Tuple[int, ...], torch.Tensor]] = [(0.0, (1, ), dec_h_c_state)]

            for dec_index in range(max_length - 1):
                curr_beam = []
                for prev_beam_prob, prev_beam_item, prev_beam_hidden_state in prev_beam:
                    prev_item_index = prev_beam_item[-1]
                    if prev_item_index in (2, 0):
                        curr_beam.append((prev_beam_prob, prev_beam_item, prev_beam_hidden_state))
                    else:
                        dec_input = torch.tensor([[prev_item_index]])
                        dec_output, dec_h_c_state = self.decoder.forward(dec_input, prev_beam_hidden_state, enc_outputs)
                        top_values, top_indices = dec_output.data.topk(num_expansions)

                        for prob, candidate in zip(top_values, top_indices):
                            curr_beam.append((prev_beam_prob + float(prob), prev_beam_item + (int(candidate), ), dec_h_c_state))
                prev_beam = []
                prev_beam_set = set()
                # print(len(curr_beam))
                for prob, item, hidden in curr_beam:
                    if (prob, item) not in prev_beam_set:
                        prev_beam_set.add((prob, item))
                        prev_beam.append((prob, item, hidden))
                # print(len(prev_beam))
                prev_beam = sorted(prev_beam, key=lambda x: x[0] / len(x[1]), reverse=True)[:beam_size]
                # print(len(prev_beam))
            return [(prob/len(seq), seq) for prob, seq, _ in prev_beam]

    def generate_greedy(self, enc_emb: torch.Tensor, max_length=50):
        with torch.no_grad():
            return self.forward(enc_emb, max_length)
