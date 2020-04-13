import numpy as np
import torch
from torch import nn as nn

from data_util import generate_dataset, generate_random_sequence


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_key_padding_mask=None):
        raise NotImplementedError('Must extend this class and implement this method on child class.')

    def decode(self, src, encoded_repr=None, initial_hidden=None, train=True, cuda=True):
        raise NotImplementedError('Must extend this class and implement this method on child class.')

    def generate_from_hidden(self, max_len, encoded_repr=None, hidden_state=None, train=True, cuda=True):
        if encoded_repr is not None:
            assert hidden_state is None
            batch_size = encoded_repr.size(0)
        else:
            assert hidden_state is not None and len(hidden_state) == 2
            batch_size = hidden_state[0].size(1)
        input_token = torch.LongTensor(np.array([[self.sos_idx]] * batch_size))
        outputs_tokens = []
        output_probabilities = []
        for li in range(max_len):
            if cuda:
                input_token = input_token.cuda()
            output, hidden_state = self.decode(
                src=input_token, encoded_repr=encoded_repr,
                initial_hidden=hidden_state, train=train, cuda=False
            )
            output_probabilities.append(output)
            input_token = torch.argmax(output, dim=-1)
            outputs_tokens.append(input_token)
            pass
        final_output_probabilities = torch.cat(output_probabilities, dim=1)
        final_output_tokens = torch.cat(outputs_tokens, dim=1)
        return final_output_probabilities, final_output_tokens

    def forward(self, src, src_key_padding_mask=None, teacher_input=True, train=True, cuda=True):
        """
        Shape:
            - src: :math:`(N, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
        """
        memory, encoded_repr, hidden_state = self.encode(src=src, src_key_padding_mask=src_key_padding_mask)
        if teacher_input:
            output, _ = self.decode(
                src=src[:, :-1], encoded_repr=encoded_repr, initial_hidden=hidden_state, train=train, cuda=cuda)
        else:
            output, _ = self.generate_from_hidden(
                max_len=src.size(1) - 1, encoded_repr=encoded_repr,
                hidden_state=hidden_state, train=train, cuda=cuda
            )
        return output


class RnnAutoEncoderWithAttn(AutoEncoder):
    def __init__(self, vocab_size, sos_idx, eos_idx, pad_idx, unk_idx,
                 d_model=512, dropout=0.1, word_dropout_rate=0.):
        super(RnnAutoEncoderWithAttn, self).__init__()
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout_rate = word_dropout_rate
        self.inp_embedding = nn.Embedding(vocab_size, d_model, self.pad_idx)
        self.inp_emb_dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=2,
            bias=True, bidirectional=True, dropout=dropout, batch_first=True
        )
        self.combiner = Attention(2 * d_model)
        self.hid_transform = nn.Linear(in_features=2 * d_model, out_features=d_model)
        self.decoder = nn.LSTM(
            input_size=2*d_model, hidden_size=d_model, num_layers=2,
            bias=True, bidirectional=True, dropout=dropout, batch_first=True
        )
        self.de_embedding = nn.Linear(2 * d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self._reset_parameters()
        self.d_model = d_model

    def encode(self, src, src_key_padding_mask=None):
        inp_embedded = self.inp_embedding(src)
        inp_embedded = self.inp_emb_dropout(inp_embedded)
        memory, _ = self.encoder(inp_embedded)
        encoded_repr, attn = self.combiner(memory, src_key_padding_mask)
        encoded_repr = self.hid_transform(encoded_repr)
        return memory, encoded_repr, None

    def decode(self, src, encoded_repr=None, initial_hidden=None, train=True, cuda=True):
        if self.word_dropout_rate > 0 and train:
            prob = torch.rand(src.size())
            if cuda:
                prob = prob.cuda()
            prob[(src.data - self.sos_idx) * (src.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = src.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            decoder_input_embedding = self.inp_embedding(decoder_input_sequence)
        else:
            decoder_input_embedding = self.inp_embedding(src)
        hidden_emb = torch.stack([encoded_repr] * decoder_input_embedding.size(1), dim=1)
        decoder_input_embedding = torch.cat((decoder_input_embedding, hidden_emb), dim=-1)
        if initial_hidden is not None:
            output, next_hidden = self.decoder(decoder_input_embedding, initial_hidden)
        else:
            output, next_hidden = self.decoder(decoder_input_embedding)
        output = self.de_embedding(output)
        output = self.softmax(output)
        return output, next_hidden


class RnnAutoEncoder(AutoEncoder):
    def __init__(self, vocab_size, sos_idx, eos_idx, pad_idx, unk_idx,
                 d_model=512, dropout=0.1, word_dropout_rate=0.):

        super(RnnAutoEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout_rate = word_dropout_rate
        self.inp_embedding = nn.Embedding(vocab_size, d_model, self.pad_idx)
        self.inp_emb_dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=2,
            bias=True, bidirectional=True, dropout=dropout, batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=2,
            bias=True, bidirectional=True, dropout=dropout, batch_first=True
        )
        self.de_embedding = nn.Linear(2 * d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self._reset_parameters()
        self.d_model = d_model

    def encode(self, src, src_key_padding_mask=None):
        inp_embedded = self.inp_embedding(src)
        inp_embedded = self.inp_emb_dropout(inp_embedded)
        memory, state = self.encoder(inp_embedded)
        return memory, None, state

    def decode(self, src, encoded_repr=None, initial_hidden=None, train=True, cuda=True):
        if self.word_dropout_rate > 0 and train:
            prob = torch.rand(src.size())
            if cuda:
                prob = prob.cuda()
            prob[(src.data - self.sos_idx) * (src.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = src.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            decoder_input_embedding = self.inp_embedding(decoder_input_sequence)
        else:
            decoder_input_embedding = self.inp_embedding(src)
        output, next_hidden = self.decoder(decoder_input_embedding, initial_hidden)
        output = self.de_embedding(output)
        output = self.softmax(output)
        return output, next_hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff = nn.Linear(in_features=hidden_dim, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, contexts, context_masks):
        """
        :param contexts: (batch_size, seq_len, n_hid)
        :param context_masks: (batch_size, seq_len)
        :return: (batch_size, n_hid)
        """
        # contexts = contexts.transpose(0, 1)
        out = self.ff(contexts)
        out = out.squeeze(dim=-1)
        masked_out = out.masked_fill(context_masks, float('-inf'))
        attn_weights = self.softmax(masked_out)
        out = attn_weights.unsqueeze(1).bmm(contexts)
        out = out.squeeze(1)
        return out, attn_weights


if __name__ == '__main__':
    batch_size = 8
    src_length = 50
    vocab_size = 1000
    d_model = 123
    sos_idx, eos_idx, pad_idx, unk_idx = 0, 1, 2, 3
    src_input, mask = generate_random_sequence(
        batch_size, vocab_size, src_length, sos_idx, eos_idx, pad_idx, unk_idx
    )
    src_input = torch.LongTensor(src_input)
    mask = torch.BoolTensor(mask)
    # loss_func = nn.NLLLoss(reduction='none')
    model = RnnAutoEncoderWithAttn(
        vocab_size, sos_idx, eos_idx, pad_idx,
        unk_idx, d_model, word_dropout_rate=0.0
    )

    output = model.forward(src=src_input, src_key_padding_mask=mask, teacher_input=False, train=False, cuda=False)
    print(output.shape)
    # memory, encoded_repr, hidden_state = model.encode(src_input, mask)
    # _tokens, _probs = model.generate_from_hidden(
    #     max_len=src_length-1, encoded_repr=encoded_repr, hidden_state=hidden_state, cuda=False)
    # print(_tokens.shape, _probs.shape)