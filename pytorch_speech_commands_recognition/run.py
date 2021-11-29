from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from command_loader import CommandLoader
import numpy as np
from model import ConvNet, FcNet, RNNet
from train import train, test
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# Training settings
parser = argparse.ArgumentParser(description='ConvNets for Speech Commands Recognition')
parser.add_argument('--train_path', default='dataset/train', help='path to the train data folder')
parser.add_argument('--test_path', default='dataset/test', help='path to the test data folder')
parser.add_argument('--valid_path', default='dataset/valid', help='path to the valid data folder')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='FcNet', help='network architecture: ConvNet,FcNet,RNNet')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--patience', type=int, default=10, metavar='N',  # 5
                    help='how many epochs of no loss improvement should we wait before stop training')
# feature extraction options
parser.add_argument('--window_size', default=.02, help='window size for the stft')
parser.add_argument('--window_stride', default=.01, help='window stride for the stft')
parser.add_argument('--window_type', default='hamming', help='window type for the stft')


args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# loading data
print("loading data ...")
train_dataset = CommandLoader(args.train_path, window_size=args.window_size, window_stride=args.window_stride,
                               window_type=args.window_type)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=1, pin_memory=args.cuda, sampler=None)

valid_dataset = CommandLoader(args.valid_path, window_size=args.window_size, window_stride=args.window_stride,
                               window_type=args.window_type)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=None,
    num_workers=1, pin_memory=args.cuda, sampler=None)  # num_workers=20

test_dataset = CommandLoader(args.test_path, window_size=args.window_size, window_stride=args.window_stride,
                              window_type=args.window_type)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=None,
    num_workers=1, pin_memory=args.cuda, sampler=None)  # num_workers=20

# build model
if args.arc == 'ConvNet':
    model = ConvNet()
elif args.arc == 'FcNet':
    model = FcNet()
elif args.arc=='RNNet':
    model = RNNet()
else:
    model = ConvNet()

if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

# define optimizer
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)



import logging
from tqdm import tqdm
 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

logger = get_logger('logs/FCNlog.txt')
logger.info('start training!')

best_valid_loss = np.inf
iteration = 0
epoch = 0

# train with early stopping
for epoch in range(args.epochs):
    if iteration >= args.patience:
        print("No improvement for long. Stop training")
        break
    train(train_loader, model, optimizer,epoch, args.epochs, args.cuda, logger,args.log_interval,args.arc)
    print("Validating")
    valid_loss = test(valid_loader, model, args.cuda,logger,args.arc, verbose=False)
    if valid_loss > best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        iteration = 0
        best_valid_loss = valid_loss
        state = {
            'net': model.module if args.cuda else model,
            'acc': valid_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpointFCN'):
            os.mkdir('checkpointFCN')
        torch.save(state, './checkpointFCN/FCNepoch{}_valid_loss={:.5f}_ckpt.t7'.format(epoch,valid_loss))
        print('model saved to ./checkpointFCN/FCNepoch_{}_valid_loss={:.5f}_ckpt.t7'.format(epoch,valid_loss))

# test model
test(test_loader, model, args.cuda,logger,args.arc)
