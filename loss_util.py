import torch


def reconstruction_loss(loss_func, src, src_mask, output):
    output = output.transpose(1, 2)
    loss = loss_func(output, src[:, 1:])
    loss = loss.masked_fill(src_mask[:, 1:], 0.)
    batch_loss = loss.sum()
    return batch_loss


def calculate_loss(model, loss_func, src, src_mask, teacher_input=False, train=True, cuda=True):
    if cuda:
        src = src.cuda()
        src_mask = src_mask.cuda()
    if not train:
        with torch.no_grad():
            model.eval()
            output = model(
                src=src, src_key_padding_mask=src_mask, teacher_input=teacher_input, train=train, cuda=cuda
            )
            batch_loss = reconstruction_loss(loss_func, src, src_mask, output)
            return batch_loss
    else:
        model.train()
        output = model(
            src=src, src_key_padding_mask=src_mask, teacher_input=teacher_input, train=train, cuda=cuda
        )
        batch_loss = reconstruction_loss(loss_func, src, src_mask, output)
        return batch_loss