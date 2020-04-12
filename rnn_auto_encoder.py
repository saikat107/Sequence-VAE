import torch
import torch.nn as nn
import copy
import numpy as np


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


class RnnAutoEncoder(nn.Module):
    def __init__(self, vocab_size, sos_idx, eos_idx, pad_idx, unk_idx,
                 d_model=512, dropout=0.1, word_dropout_rate=0.1):

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

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_key_padding_mask=None):
        inp_embedded = self.inp_embedding(src)
        inp_embedded = self.inp_emb_dropout(inp_embedded)
        memory, state = self.encoder(inp_embedded)
        return memory, state

    def decode(self, src, hidden_state, train=True, cuda=True):
        decoder_src = src[:, :-1]
        if self.word_dropout_rate > 0 and train:
            prob = torch.rand(decoder_src.size())
            if cuda:
                prob = prob.cuda()
            prob[(decoder_src.data - self.sos_idx) * (decoder_src.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = decoder_src.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            decoder_input_embedding = self.inp_embedding(decoder_input_sequence)
        else:
            decoder_input_embedding = self.inp_embedding(decoder_src)
        output, next_hidden = self.decoder(decoder_input_embedding, hidden_state)
        output = self.de_embedding(output)
        output = self.softmax(output)
        return output, next_hidden

    def generate_from_hidden(self, hidden_state, max_len, train=True, cuda=True):
        batch_size = hidden_state.size(0)
        init_token = torch.LongTensor(np.array([[self.sos_idx]] * batch_size))
        if cuda:
            init_token = init_token.cuda()
        outputs = []
        for li in range(max_len):
            pass

    def forward(self, src, src_key_padding_mask=None, train=True, cuda=True):
        """
        Shape:
            - src: :math:`(N, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
        """
        memory, hidden_state = self.encode(src=src, src_key_padding_mask=src_key_padding_mask)
        output, next_state = self.decode(src=src, hidden_state=hidden_state, train=train, cuda=cuda)
        return output, hidden_state, next_state


class RnnAutoEncoderWithAttn(nn.Module):
    def __init__(self, vocab_size, sos_idx, eos_idx, pad_idx, unk_idx,
                 d_model=512, dropout=0.1, word_dropout_rate=0.1):
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
        self.decoder = nn.LSTM(
            input_size=2*d_model, hidden_size=d_model, num_layers=2,
            bias=True, bidirectional=True, dropout=dropout, batch_first=True
        )
        self.combiner = Attention(2 * d_model)
        self.hid_transform = nn.Linear(in_features=2 * d_model, out_features=d_model)
        self.de_embedding = nn.Linear(2 * d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_key_padding_mask=None):
        inp_embedded = self.inp_embedding(src)
        inp_embedded = self.inp_emb_dropout(inp_embedded)
        memory, state = self.encoder(inp_embedded)
        state, attn = self.combiner(memory, src_key_padding_mask)
        state = self.hid_transform(state)
        return memory, state

    def decode(self, src, hidden_state, train=True, cuda=True):
        decoder_src = src[:, :-1]
        if self.word_dropout_rate > 0 and train:
            prob = torch.rand(decoder_src.size())
            if cuda:
                prob = prob.cuda()
            prob[(decoder_src.data - self.sos_idx) * (decoder_src.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = decoder_src.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            decoder_input_embedding = self.inp_embedding(decoder_input_sequence)
        else:
            decoder_input_embedding = self.inp_embedding(decoder_src)
        hidden_emb = torch.stack([hidden_state] * decoder_input_embedding.size(1), dim=1)
        decoder_input_embedding = torch.cat((decoder_input_embedding, hidden_emb), dim=-1)
        output, state = self.decoder(decoder_input_embedding)
        output = self.de_embedding(output)
        output = self.softmax(output)
        return output, state

    def generate_from_hidden(self, hidden_state, max_len, train=True, cuda=True):
        batch_size = hidden_state.size(0)
        init_token = torch.LongTensor(np.array([[self.sos_idx]] * batch_size))
        if cuda:
            init_token = init_token.cuda()
        outputs = []
        for li in range(max_len):
            pass

    def forward(self, src, src_key_padding_mask=None, train=True, cuda=True):
        """
        Shape:
            - src: :math:`(N, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
        """
        memory, hidden_state = self.encode(src=src, src_key_padding_mask=src_key_padding_mask)
        output, next_state = self.decode(src=src, hidden_state=hidden_state, train=train, cuda=cuda)
        return output, hidden_state, next_state


def reconstruction_loss(loss_func, src, src_mask, output):
    output = output.transpose(1, 2)
    loss = loss_func(output, src[:, 1:])
    loss = loss.masked_fill(src_mask[:, :-1], 0.)
    batch_loss = loss.sum()
    return batch_loss


def calculate_loss(model, loss_func, src, src_mask, train=True, cuda=True):
    if cuda:
        src = src.cuda()
        src_mask = src_mask.cuda()
    if not train:
        with torch.no_grad():
            model.eval()
            output, _, _ = model(src, src_mask, train, cuda)
            batch_loss = reconstruction_loss(loss_func, src, src_mask, output)
            return batch_loss
    else:
        model.train()
        output, _, _ = model(src, src_mask, train, cuda)
        batch_loss = reconstruction_loss(loss_func, src, src_mask, output)
        return batch_loss


def generate_random_sequence(batch_size, vocab_size, src_length, sos_idx, eos_idx, pad_idx, unk_idx):
    input = np.random.randint(unk_idx + 1, vocab_size + 2, size=(batch_size, src_length))
    masks = np.zeros(shape=(batch_size, src_length))
    for i in range(batch_size):
        for j in range(src_length):
            if input[i, j] >= vocab_size:
                input[i, j] = unk_idx
    lengths = np.random.randint(int(src_length/2), src_length, size=batch_size)
    for i in range(batch_size):
        l = lengths[i]
        input[i, 0] = sos_idx
        input[i, l-1] = eos_idx
        input[i, l:] = pad_idx
        masks[i, l:] = 1
    print(vocab_size, np.max(input))
    return input, masks
    pass


def generate_dataset(b_size, v_size, s_len, s_idx, e_idx, p_idx, u_idx):
    batches = []
    for _ in range(num_batches):
        src, src_mask = generate_random_sequence(
            b_size, v_size, s_len, s_idx, e_idx, p_idx, u_idx)
        src = torch.LongTensor(src)
        src_mask = torch.BoolTensor(src_mask)
        batches.append((src, src_mask))
    train_batches = batches[:70]
    valid_batches = batches[70:80]
    test_batches = batches[80:]
    return train_batches, valid_batches, test_batches


if __name__ == '__main__':
    batch_size = 32
    src_length = 200
    vocab_size = 10000
    d_model = 256
    num_batches = 100
    num_epochs = 400
    sos_idx, eos_idx, pad_idx, unk_idx = 0, 1, 2, 3
    train_batches, valid_batches, test_batches = generate_dataset(
        batch_size, vocab_size, src_length, sos_idx, eos_idx, pad_idx, unk_idx
    )
    loss_func = nn.NLLLoss(reduction='none')
    model = RnnAutoEncoderWithAttn(
        vocab_size, sos_idx, eos_idx, pad_idx,
        unk_idx, d_model, word_dropout_rate=0.0
    )
    print(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    model.zero_grad()
    optimizer.zero_grad()
    model.train()
    train_losses = []
    valid_losses = []
    test_losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_model = None
    for i in range(num_epochs):
        try:
            if i % 1 == 0:
                valid_batch_loss = []
                for bi, (src, src_mask) in enumerate(valid_batches):
                    loss = calculate_loss(
                        model, loss_func, src, src_mask, train=False, cuda=True
                    )
                    loss_val = loss.cpu().item()
                    valid_batch_loss.append(loss_val)
                valid_loss = np.mean(valid_batch_loss)
                valid_losses.append(valid_loss)

                test_batch_loss = []
                for bi, (src, src_mask) in enumerate(test_batches):
                    loss = calculate_loss(
                        model, loss_func, src, src_mask, train=False, cuda=True
                    )
                    loss_val = loss.cpu().item()
                    test_batch_loss.append(loss_val)
                test_loss = np.mean(test_batch_loss)
                test_losses.append(test_loss)
                if valid_loss < best_loss:
                    patience_counter = 0
                    best_model = model.state_dict()
                    best_loss = valid_loss
                else:
                    patience_counter += 1

                if patience_counter == 20:
                    if best_model is not None:
                        print(model.load_state_dict(best_model))
                    break
            batch_loss = []
            for bi, (src, src_mask) in enumerate(train_batches):
                model.zero_grad()
                optimizer.zero_grad()
                loss = calculate_loss(
                    model, loss_func, src, src_mask, train=True, cuda=True
                )
                loss.backward()
                optimizer.step()
                loss_val = loss.cpu().item()
                batch_loss.append(loss_val)
            print('Train:\tepoch %d\tLoss: %.5f'% (i, float(np.mean(batch_loss))))
            print(
                'Valid\tepoch %d\tLoss: %.5f\tPatience: %d' %
                (i, float(valid_loss), patience_counter)
            )
            print(
                'Test:\tepoch %d\tLoss: %.5f' %
                (i, float(test_loss))
            )
            train_losses.append(np.mean(batch_loss))
        except KeyboardInterrupt:
            print('Training stopped by user!')
            break
    from matplotlib import pyplot as plt
    plt.figure('Losses_upto_10it')
    plt.title('Losses_upto_10it')
    plt.plot(train_losses[:10], 'r-', label='Train')
    plt.plot(valid_losses[:10], 'b-', label='Valid')
    plt.plot(test_losses[:10], 'g-', label='Test')
    plt.legend()
    plt.savefig('Losses_upto_10it_attn.png')

    plt.figure('Losses_from_10_to_20_it')
    plt.title('Losses_from_10_to_20_it')
    plt.plot(train_losses[10:20], 'r-', label='Train')
    plt.plot(valid_losses[10:20], 'b-', label='Valid')
    plt.plot(test_losses[10:20], 'g-', label='Test')
    plt.legend()
    plt.savefig('Losses_from_10_to_20_it_attn.png')

    plt.figure('Losses_it_20_till_end')
    plt.title('Losses_it_20_till_end')
    plt.plot(train_losses[20:], 'r-', label='Train')
    plt.plot(valid_losses[20:], 'b-', label='Valid')
    plt.plot(test_losses[20:], 'g-', label='Test')
    plt.legend()
    plt.savefig('Losses_it_20_till_end_attn.png')
    plt.show()
