import os
import torch
import argparse
import datetime

import torch.nn.functional as F
import torchtext.data as data
import _pickle as pickle
import numpy as np

from torch.autograd import Variable
from dataset import dataset
from model import ContextWordEmb
from sklearn.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser(description="Context Word Embedding")
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-cuda', type=bool, default=True)
parser.add_argument('-epochs', type=int, default=25, help='number of epochs for train [default: 25]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 16]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

def load_data(text_field, char_field, label_field, **kwargs):
    print("read dataset...")
    train_data, test_data = dataset.splits(text_field, char_field, label_field)
    print("build word vocabulary...")
    text_field.build_vocab(train_data, min_freq=2)
    print("build character vocabulary...")
    label_field.build_vocab(train_data, test_data)
    char_field.build_vocab(train_data)
    print("build data iterator...")
    train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data),
        batch_sizes=(args.batch_size, args.batch_size * 2),
        shuffle=args.shuffle,
        **kwargs
    )
    return train_iter, test_iter

def train(model, train_iter, test_iter):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    model.train()
    n_batch = len(train_iter)
    step = 0
    for epoch in range(args.epochs):
        scores = np.array([])
        target = np.array([])
        for batch in train_iter:
            (sent, char), labels = (batch.text, batch.char), batch.label
            sent = (sent.cuda(), char.cuda())
            batch_size = labels.size()[0]
            labels = labels.cuda()
            sent_rep = model(sent)

            logits = model(sent_rep, labels)

            optimizer.zero_grad()
            loss = F.cross_entropy(F.softmax(logits, 1), labels)
            loss = torch.sum(loss) / batch_size
            loss.backward()
            optimizer.step()

            predict = torch.max(logits, 1)[1].squeeze()
            scores = np.append(scores, predict.cpu().data.numpy())
            target = np.append(target, labels.cpu().data.numpy())

            step += 1

            if step % (1000 * 32 / model.batch_size) == 0:
                acc = accuracy_score(target, scores)
                print('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Acc: %.4f'
                      % (epoch + 1, model.epochs, step, n_batch * model.epochs, loss.data[0], acc))
                scores = np.array([])
                target = np.array([])

        test_acc = evaluate(model, test_iter)
        if not os.path.isdir(model.save_dir): os.makedirs(model.save_dir)
        save_prefix = os.path.join(model.save_dir, 'snapshot')
        save_path = '{}_steps_{}_{}.pt'.format(save_prefix, step, test_acc)
        torch.save(model, save_path)


def evaluate(model, test_iter):
    scores = np.array([])
    target = np.array([])
    for batch in test_iter:
        (sent, char), labels = (batch.text, batch.char), batch.label
        sent = (Variable(sent.data, volatile=True).cuda(), Variable(char.data, volatile=True).cuda())

        labels = labels.cuda()
        batch_size = labels.size()[0]

        sent_rep = model(sent)
        logits = model.classifier(sent_rep.view(batch_size, -1))

        logits = F.softmax(logits, 1)
        predict = torch.max(logits, 1)[1].squeeze()

        scores = np.append(scores, predict.cpu().data.numpy())
        target = np.append(target, labels.cpu().data.numpy())

    acc = accuracy_score(target, scores)
    confusion_mat = classification_report(target, scores, target_names=label_field.vocab.itos)
    print('Test Performance of the model: Acc: %.4f' % (acc))
    print(confusion_mat)
    return int(1000 * acc)

if __name__ == "__main__":

    print("\nLoading data...")
    text_field = data.Field(lower=True, batch_first=True)
    label_field = data.Field(sequential=False, unk_token=None)
    train_iter, test_iter = load_data(text_field, label_field, device=args.device, repeat=False)

    args.vocab_size = len(text_field.vocab)
    args.n_class = len(label_field.vocab)
    args.dictionary = text_field.vocab

    print("\nSaving dictionary...")
    with open('word_dictoinary.pkt', 'wb') as handle:
        pickle.dump(text_field.vocab, handle)

    args.cuda = (not args.no_cuda) and torch.cuda.is_available();
    del args.no_cuda
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # model
    if args.snapshot is None:
        context_word_classifier = ContextWordEmb(args)
    else:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            context_word_classifier = torch.load(args.snapshot)
        except:
            print("Sorry, This snapshot doesn't exist.");
            exit()

    if args.cuda:
        context_word_classifier = context_word_classifier.cuda()

    # train
    train(context_word_classifier, train_iter, test_iter)