import os
import torch


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def moving_average(net1, net2, ema_decay=0.9999):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data = param1.data * (1.0 - ema_decay) + param2.data * ema_decay


def all_average(net_list):
    net_base = net_list[0]
    net_number = len(net_list)
    for i in range(1, net_number):
        for param1, param2 in zip(net_base.parameters(), net_list[i].parameters()):
            param1.data = param1.data + param2.data
    for param in net_base.parameters():
        param.data = param.data / net_number
    return net_base


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, amp):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    model = torch.nn.DataParallel(model).cuda().train()
    if not check_bn(model):
        return
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum
        if amp:
            with torch.cuda.amp.autocast():
                model(input)
        else:
            model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    return model
