# -*- coding: utf-8 -*-
"""
@author: RMSnow
@file: eval.py
@time: 2020/10/29 10:50
@contact: xueyao_98@foxmail.com

"""

import os
import time
import json
from tqdm import tqdm
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW, get_cosine_schedule_with_warmup

from RougeBert import RougeBert
from DatasetLoader import DatasetLoader
from config import parser


def evaluate(args, loader, model, criterion, type):
    print('Eval time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    model.eval()
    total_loss = 0.
    outputs = []
    num_batches = 0

    with torch.no_grad():
        for step, (qidxs, didxs, sidxs, labels) in enumerate(tqdm(loader)):
            # (bz, 2)
            output = model(qidxs, didxs, sidxs)
            labels = torch.cat([x[:, None] for x in labels], dim=-1)
            labels = torch.as_tensor(
                labels, dtype=torch.float, device=args.device)

            loss = criterion(output, labels)
            total_loss += loss.item()

            for i, qidx in enumerate(qidxs):
                qidx = qidx.item()
                didx = didxs[i].item()
                sidx = sidxs[i].item()
                outputs.append(
                    (qidx, didx, sidx, output[i].cpu().numpy().tolist()))

            num_batches += 1

    e = args.current_epoch
    total_loss /= num_batches
    file = os.path.join(args.save, type + '_outputs_' + str(e) + '_' + str(
        args.local_rank) + '_loss_{:.4f}'.format(total_loss) + '.json')
    json.dump(outputs, open(file, 'w'))

    return total_loss


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
    model = RougeBert(args)
    print(model)
    print('\nLoading model time: {:.2f}s\n-----------------------------------------\n'.format(
        time.time() - start))

    criterion = nn.MSELoss().cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr)

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
    train_dataset = DatasetLoader('train', nrows=nrows, dataset=args.dataset)
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

    val_sampler = RandomSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=val_sampler
    )

    test_sampler = RandomSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
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

    layers = []
    for name in model.state_dict():
        layers.append(name)
    init_model_params = model.state_dict()

    best_val_loss = None
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch
        print('\n------------------------------------------------\n')
        print('Start Training Epoch', epoch, ':', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        model.train()

        train_loss = 0
        lr = optimizer.param_groups[0]['lr']
        for step, (qidxs, didxs, sidxs, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            with autocast():
                # labels: two elements. every element's shape = tensor([bz])
                # so: two (bz, 1) -> (bz, 2)
                labels = torch.cat([x[:, None] for x in labels], dim=-1)
                labels = torch.as_tensor(
                    labels, dtype=torch.float, device=args.device)

                output = model(qidxs, didxs, sidxs)
                loss = criterion(output, labels)

                reg_loss = 0
                for layer in layers:
                    if 'weight' in layer or 'bias' in layer:
                        diff = init_model_params[layer] - \
                            model.state_dict()[layer]
                        reg_loss += diff.norm(2) ** 2

                loss += args.rouge_bert_regularize * reg_loss

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
                os.path.join(args.save, '{}.pt'.format(epoch))
            )

        print('Epoch [{}/{}]\t Train Loss: {}\t Val Loss: {}\t Test Loss: {}\t lr: {}\n'.format(
            epoch, args.epochs, train_loss, val_loss, test_loss, lr))

    print('Training Time:', int(time.time() - start))
    print('End time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
