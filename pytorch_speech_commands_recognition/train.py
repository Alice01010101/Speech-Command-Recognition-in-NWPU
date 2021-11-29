from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
from tqdm import tqdm

def train(loader, model, optimizer, epoch,epoch_num, cuda, logger,log_interval,arc, verbose=True):
    '''
    #############################################################################
    train the model, you can write this function partly refer to the "test" below
    Args:
        loader: torch dataloader
        model: model to train
        optimizer: torch optimizer
        epoch: number of epochs to train
        cuda: whether to use gpu (True or False)
        log_interval: how many batches to wait before logging training status
        verbose: whether to print training log(such as epoch and loss)
    Return:
        the average loss of this epoch
    #############################################################################
    '''
    model.train() #用于设置BN和Dropout的状态
    train_loss=0
    correct=0
    p_bar = tqdm(loader, desc='Epoch %i' % epoch)

    for data, target in loader:
        if cuda:
            data, target=data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target) # volatile=True的节点不会求导，即使requires_grad=True，也不会进行反向传播
        with torch.no_grad():
            data = Variable(data)
        target = Variable(target)

        if arc=='RNNet':
            output,hidden=model(data) #output[16100,32] hidden[2,100,32]
        else:
            output = model(data)

        loss= F.nll_loss(F.log_softmax(output,dim=-1), target, reduction='sum')  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        #back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data.item()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_loss/=len(loader.dataset)

    if verbose:
        logger.info('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    logger.info('finish training for epoch {}!'.format(epoch))

    return train_loss


def test(loader, model, cuda, logger, arc,verbose=True):
    model.eval() #用于设置BN和Dropout状态
    test_loss = 0
    correct = 0
    p_bar = tqdm(loader)
    for data, target in p_bar:
        if cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target)
        with torch.no_grad():
            data = Variable(data)
        target = Variable(target)

        if arc=='RNNet':
            output,hidden=model(data) #output[16100,32] hidden[2,100,32]
        else:
            output = model(data)
            
        test_loss += F.nll_loss(F.log_softmax(output,dim=-1), target, reduction='sum').data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        p_bar.set_postfix(
            avg_loss=test_loss/ loader.batch_size,
            acc=correct.item() / loader.batch_size
        )

    test_loss /= len(loader.dataset)
    
    #当在验证集上进行验证的时候，将verbose设置为false
    if verbose:
        logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    else:
        logger.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss
