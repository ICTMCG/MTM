import os
import time
import json
from tqdm import tqdm
import pickle
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW, get_cosine_schedule_with_warmup

from MTM import MTM
from DatasetLoader import DatasetLoader
from config import parser
from evaluation import eval_for_outputs


def evaluate(args, loader, model, criterion, type):
    print('Eval time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    model.eval()
    total_loss = 0.
    outputs = []

    if type == 'train_eval':
        with torch.no_grad():
            for qids, qidxs, dids, didxs, labels in tqdm(loader):
                output = model(qidxs, didxs)
                loss = criterion(output, labels.to(args.device))
                total_loss += loss.item()

                # save predictions for updating memory
                output_prob = softmax(output)
                predictions = [(qidxs[i], didxs[i], t, output_prob[i][t])
                               for i, t in enumerate(labels)]
                predictions = [[x.item() for x in p] for p in predictions]

                outputs += predictions

        total_loss /= len(loader)
        file = os.path.join(args.save, type + '_outputs_' +
                            str(args.current_epoch) + '.json')
        json.dump(outputs, open(file, 'w'))

        return total_loss, file

    elif type in ['val', 'test', 'val_updated', 'test_updated']:
        with torch.no_grad():
            for qid, qidx, dids, didxs, labels in tqdm(loader):
                output = model([qidx] * len(didxs), didxs)
                loss = criterion(output, torch.as_tensor(labels).to(args.device))
                total_loss += loss.item()

                try:
                    score = [(dids[i][0].item(), x[0], x[1])
                             for i, x in enumerate(output.cpu().numpy().tolist())]
                except:
                    score = [(dids[i][0], x[0], x[1])
                             for i, x in enumerate(output.cpu().numpy().tolist())]

                try:
                    outputs.append((qid[0].item(), score))
                except:
                    outputs.append((qid[0], score))

        total_loss /= len(loader)
        file = os.path.join(args.save, type + '_outputs_' +
                            str(args.current_epoch) + '.json')
        json.dump(outputs, open(file, 'w'))

        eval_for_outputs(typ=type, dataset=args.dataset, outputs_file=file)
        return total_loss

    else:
        print('Error: the illegal param of evaluate_type!')
        exit()


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        args.save = './ckpts/debug'
        args.epochs = 2

    if os.path.exists(args.save):
        os.system('rm -r {}'.format(args.save))
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    print('\n{} Experimental Dataset: {} {}\n'.format(
        '=' * 20, args.dataset, '=' * 20))
    print('save path: ', args.save, '\n')
    print('Start time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('-----------------------------------------\nLoading model...\n')
    start = time.time()
    model = MTM(args)
    print(model)
    print('\nLoading model time: {:.2f}s\n-----------------------------------------\n'.format(
        time.time() - start))

    criterion = nn.CrossEntropyLoss().cuda()
    softmax = nn.Softmax(dim=1)
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                             model.parameters()), lr=args.lr)

    if args.fp16:
        scaler = GradScaler()

    if torch.cuda.is_available():
        model = model.cuda()

    if args.resume != '':
        resume_dict = torch.load(args.resume)

        model.load_state_dict(resume_dict['state_dict'])
        optimizer.load_state_dict(resume_dict['optimizer'])
        args.start_epoch = resume_dict['epoch'] + 1

    print('Loading data...')
    start = time.time()

    nrows = 20 * args.batch_size if args.debug else None
    train_dataset = DatasetLoader(
        'train.line', nrows=nrows, dataset=args.dataset)
    nrows = 20 if args.debug else None
    val_dataset = DatasetLoader('val', nrows=nrows, dataset=args.dataset)
    test_dataset = DatasetLoader('test', nrows=nrows, dataset=args.dataset)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=train_sampler
    )

    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=val_sampler
    )

    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=test_sampler
    )

    print('Loading data time: {}s'.format(int(time.time() - start)))

    if args.resume == '':
        args_file = os.path.join(args.save, 'args.txt')
    else:
        resume_from = args.resume.split('/')[-1]
        args_file = os.path.join(
            args.save, 'args_resume_from_{}.txt'.format(resume_from))
    with open(args_file, 'w') as f:
        print_s = ''
        for arg in vars(args):
            s = '{}\t{}\n'.format(arg, getattr(args, arg))
            f.write(s)
            print_s += s

        print('\n---------------------------------------------------\n')
        print('[Arguments] \n')
        print(print_s)
        print('\n---------------------------------------------------\n')

    last_epoch = args.start_epoch if args.start_epoch != 0 else -1
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(train_loader) * (args.epochs - args.start_epoch + 1),
                                                last_epoch)

    # Training
    print('Start training...')
    start = time.time()
    args.global_step = 0

    best_val_loss = None
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch
        print('\n------------------------------------------------\n')
        print('Start Training Epoch', epoch, ':', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        model.train()

        train_loss = 0
        lr = optimizer.param_groups[0]['lr']
        for step, (qids, qidxs, dids, didxs, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            with autocast():
                output = model(qidxs, didxs)
                loss = criterion(output, labels.to(args.device))

            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if scheduler:
                scheduler.step()

            train_loss += loss.item()
            args.global_step += 1

        train_loss /= len(train_loader)
        val_loss = evaluate(args, val_loader, model, criterion, type='val')
        test_loss = evaluate(args, test_loader, model, criterion, type='test')

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                os.path.join(
                    args.save, '{}_before_memory_updated.pt'.format(epoch))
            )

        train_eval_loss, train_eval_file = evaluate(
            args, train_loader, model, criterion, 'train_eval')
        model.update_memory_after_epoch(train_eval_file, epoch)
        val_loss_updated = evaluate(
            args, val_loader, model, criterion, 'val_updated')
        test_loss_updated = evaluate(
            args, test_loader, model, criterion, 'test_updated')

        if best_val_loss > val_loss_updated:
            best_val_loss = val_loss_updated
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                os.path.join(args.save, '{}.pt'.format(epoch))
            )

        print('Epoch [{}/{}]\t Train Loss: {}\t Val Loss: {}\t Test Loss: {}\t Val Updated Loss: {}\t Test Updated Loss: {}\t lr: {}\n'.format(
            epoch, args.epochs, train_eval_loss, val_loss, test_loss, val_loss_updated, test_loss_updated, lr))

    print('Training Time:', int(time.time() - start))
    print('End time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
