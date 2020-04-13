import numpy as np
import torch


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
    return input, masks
    pass


def generate_dataset(num_batches, b_size, v_size, s_len, s_idx, e_idx, p_idx, u_idx):
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