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

    if args.debug:
        train_dataset = DatasetLoader(
            'train', nrows=20 * args.batch_size, dataset=args.dataset)
    else:
        train_dataset = DatasetLoader('train', dataset=args.dataset)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=train_sampler
    )

    with open('./data/{}/rouge2_labels.pkl'.format(args.dataset), 'rb') as f:
        rouge_labels = pickle.load(f)

    print('Loading data time:', int(time.time() - start))

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

        print(f"Epoch [{epoch}/{args.epochs}]\t \
                Train Loss: {train_loss / len(train_loader)}\t \
                lr: {round(lr, 5)}")

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        },
            os.path.join(args.save, '{}.pt'.format(epoch))
        )

    print('Training Time:', int(time.time() - start))
    print('End time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
