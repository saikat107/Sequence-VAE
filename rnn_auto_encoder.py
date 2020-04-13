import torch
import torch.nn as nn
import numpy as np

from data_util import generate_dataset
from loss_util import calculate_loss
from modules import RnnAutoEncoderWithAttn, RnnAutoEncoder
from datetime import datetime


def accumulate_losses(model, batches, loss_func, teacher_input_prob, train, cuda):
    _batch_loss = []
    for bi, (src, src_mask) in enumerate(batches):
        if train:
            model.zero_grad()
            optimizer.zero_grad()
        prob = np.random.uniform(0, 1)
        teacher_input = prob <= teacher_input_prob
        loss = calculate_loss(
            model, loss_func, src, src_mask, teacher_input=teacher_input, train=train, cuda=cuda
        )
        if train:
            loss.backward()
            optimizer.step()
        loss_val = loss.cpu().item()
        _batch_loss.append(loss_val)
    _loss = float(np.mean(_batch_loss))
    return _loss


if __name__ == '__main__':
    np.random.seed(123456)
    batch_size, src_length, vocab_size, d_model, num_batches = 32, 200, 5000, 256, 100
    num_epochs, max_patience, patience_counter = 300, 20, 0
    best_model, best_loss = None, float('inf')
    cuda = torch.cuda.is_available()
    teacher_input_probability = 1.
    sos_idx, eos_idx, pad_idx, unk_idx = 0, 1, 2, 3
    train_batches, valid_batches, test_batches = generate_dataset(
        num_batches, batch_size, vocab_size,
        src_length, sos_idx, eos_idx, pad_idx, unk_idx
    )
    loss_func = nn.NLLLoss(reduction='none')
    model = RnnAutoEncoderWithAttn(
        vocab_size, sos_idx, eos_idx, pad_idx,
        unk_idx, d_model, word_dropout_rate=0.0
    )
    print(model)
    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    model.zero_grad()
    optimizer.zero_grad()
    model.train()
    train_losses, valid_losses, test_losses = [], [], []
    for i in range(num_epochs):
        try:
            valid_loss = accumulate_losses(model=model, batches=valid_batches, loss_func=loss_func,
                                           teacher_input_prob=teacher_input_probability, train=False, cuda=cuda)
            valid_losses.append(valid_loss)
            test_loss = accumulate_losses(model=model, batches=test_batches, loss_func=loss_func,
                                          teacher_input_prob=teacher_input_probability, train=False, cuda=cuda)
            test_losses.append(test_loss)

            if valid_loss < best_loss:
                patience_counter = 0
                best_model = model.state_dict()
                best_loss = valid_loss
            else:
                patience_counter += 1
            if patience_counter == max_patience:
                if best_model is not None:
                    model.load_state_dict(best_model)
                break

            train_loss = accumulate_losses(model=model, batches=train_batches, loss_func=loss_func,
                                           teacher_input_prob=teacher_input_probability, train=True, cuda=cuda)
            train_losses.append(train_loss)
            print('[%s]\tNormal Epoch %3d\tTrain: %8.2f\tValid: %8.2f\tTest: %8.2f\tPatience: %d' \
                  % (str(datetime.now()), i, train_loss, valid_loss, test_loss, patience_counter))

        except KeyboardInterrupt:
            print('Training stopped by user!')
            break

    test_loss = accumulate_losses(model=model, batches=test_batches, loss_func=loss_func,
                                  teacher_input_prob=0., train=False, cuda=cuda)
    print('Best Test Loss Found: %5.5f' % test_loss)
    from matplotlib import pyplot as plt

    plt.title('Losses')
    plt.plot(train_losses, 'r-', label='Train')
    plt.plot(valid_losses, 'b-', label='Valid')
    plt.plot(test_losses, 'g-', label='Test')
    plt.legend()
    plt.savefig('visuals/Normal-Losses.png')
    plt.show()

    print('=' * 100)
    print('Starting Fine Tune')
    print('=' * 100)
    train_losses, valid_losses, test_losses = [], [], []
    best_loss = float('inf')
    for i in range(num_epochs):
        try:
            valid_loss = accumulate_losses(model=model, batches=valid_batches, loss_func=loss_func,
                                           teacher_input_prob=0., train=False, cuda=cuda)
            valid_losses.append(valid_loss)
            test_loss = accumulate_losses(model=model, batches=test_batches, loss_func=loss_func,
                                          teacher_input_prob=0., train=False, cuda=cuda)
            test_losses.append(test_loss)

            if valid_loss < best_loss:
                patience_counter = 0
                best_model = model.state_dict()
                best_loss = valid_loss
            else:
                patience_counter += 1
            if patience_counter == max_patience:
                if best_model is not None:
                    model.load_state_dict(best_model)
                break

            train_loss = accumulate_losses(model=model, batches=train_batches, loss_func=loss_func,
                                           teacher_input_prob=0., train=True, cuda=cuda)
            train_losses.append(train_loss)
            print('[%s]\tFTune Epoch %3d\tTrain: %9.2f\tValid: %9.2f\tTest: %9.2f\tPatience: %d' \
                  % (str(datetime.now()), i, train_loss, valid_loss, test_loss, patience_counter))

        except KeyboardInterrupt:
            print('Training stopped by user!')
            break
    test_loss = accumulate_losses(model=model, batches=test_batches, loss_func=loss_func,
                                  teacher_input_prob=0., train=False, cuda=cuda)
    print('Best Test Loss Found: %5.5f' % test_loss)
    from matplotlib import pyplot as plt

    plt.title('Losses')
    plt.plot(train_losses, 'r-', label='Train')
    plt.plot(valid_losses, 'b-', label='Valid')
    plt.plot(test_losses, 'g-', label='Test')
    plt.legend()
    plt.savefig('visuals/Fine-tune-Losses.png')
    plt.show()
