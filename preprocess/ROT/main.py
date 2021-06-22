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

import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast

from transformers import AdamW, get_cosine_schedule_with_warmup

from RougeBert import RougeBert
from dataset_cn import DatasetCN
from config import parser


def evaluate(args, loader, model, criterion, type):
    if args.local_rank in [-1, 0]:
        print('Eval time:', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    model.eval()
    total_loss = 0.
    outputs = []
    with torch.no_grad():
        for step, (qids, qidxs, dids, didxs, sidxs, labels) in enumerate(tqdm(loader)):
            # (bz, 2)
            output = model(qidxs, didxs, sidxs)
            labels = torch.cat([x[:, None] for x in labels], dim=-1)
            labels = torch.as_tensor(
                labels, dtype=torch.float, device=args.device)

            loss = criterion(output, labels.to(args.device))
            total_loss += loss.item()

            for i, qidx in enumerate(qidxs):
                qidx = qidx.item()
                didx = didxs[i].item()
                sidx = sidxs[i].item()
                outputs.append(
                    (qidx, didx, sidx, output[i].cpu().numpy().tolist()))

    if not args.evaluate:
        e = args.current_epoch
    else:
        e = args.current_epoch - 1

    file = os.path.join(args.save, type + '_outputs_' + str(args.topk) + '_' + str(e) + '_' + str(
        args.local_rank) + '_loss_{:.4f}'.format(total_loss) + '.json')
    json.dump(outputs, open(file, 'w'))

    return total_loss


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        args.save = './ckpts/debug'
        args.epochs = 2

    if args.local_rank in [0, -1]:
        print('\n{} Experimental Dataset: {} {}\n'.format(
            '=' * 20, args.dataset, '=' * 20))
        print('Start time:', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    # if args.n_gpu > 1:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #     dist.barrier()

    # # dir to save
    # if args.save == './ckpts/':
    #     dirs = os.listdir(args.save)
    #     if len(dirs) == 0:
    #         args.save += '0/'
    #     else:
    #         dirs = [int(d) for d in dirs]
    #         dirs.sort()
    #         args.save += (str(dirs[-1] + 1) + '/')

    # if args.local_rank in [0, -1]:
    # print('save path: ', args.save, '\n')

    print('save path: ', args.save, '\n')

    # if args.local_rank in [-1, 0]:
    #     writer = SummaryWriter(log_dir=args.save)

    print('-----------------------------------------\nLoading model...\n')
    start = time.time()
    model = RougeBert(args)
    print(model)
    print('\nLoading model time: {:.2f}s\n-----------------------------------------\n'.format(
        time.time() - start))

    criterion = nn.MSELoss().cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if n not in no_decay],
    #      'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in param_optimizer if n in no_decay],
    #      'weight_decay_rate': 0.0}
    # ]

    if args.fp16:
        scaler = GradScaler()

    model = model.cuda()
    # if args.local_rank != -1:
    #     model = DDP(
    #         model,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #     )

    if args.resume != '':
        # if args.n_gpu > 1:
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        #     resume_dict = torch.load(args.resume, map_location=map_location)
        # else:
        #     resume_dict = torch.load(args.resume)

        resume_dict = torch.load(args.resume)

        model.load_state_dict(resume_dict['state_dict'])
        optimizer.load_state_dict(resume_dict['optimizer'])
        args.start_epoch = resume_dict['epoch'] + 1

    # if args.local_rank in [-1, 0]:
    print('Loading data...')
    train_dataset = DatasetCN(args.topk, 'rouge.sent', dataset=args.dataset)

    if args.debug:
        train_dataset = DatasetCN(
            args.topk, 'rouge.sent', nrows=20 * args.batch_size, dataset=args.dataset)

    # if args.local_rank != -1:
    #     train_sampler = DistributedSampler(train_dataset)
    # else:
    train_sampler = RandomSampler(train_dataset)

    start = time.time()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=train_sampler
    )

    # rouge_labels = pickle.load(open('./data/rouge/rouge2_labels.pkl', 'rb'))

    # if args.local_rank in [-1, 0]:
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

    # Only evaluate
    if args.evaluate:
        if args.resume == '':
            print('No trained .pt file loaded.')
        else:
            print('Start Evaluating... local_rank=', args.local_rank)
            args.current_epoch = args.start_epoch
            eval_train_loss = evaluate(
                args, train_loader, model, criterion, 'train')
        exit()

    last_epoch = args.start_epoch if args.start_epoch != 0 else -1
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(train_loader) * (args.epochs - args.start_epoch + 1),
                                                last_epoch)

    # Training
    # if args.local_rank in [-1, 0]:
    print('Start training...')

    start = time.time()
    args.global_step = 0

    layers = []
    for name in model.state_dict():
        layers.append(name)
    init_model_params = model.state_dict()

    print('=' * 25)
    print('\nRegularize parameter: {}\n'.format(args.rouge_bert_regularize))
    print('=' * 25)

    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch
        # if args.local_rank in [-1, 0]:
        print('\n------------------------------------------------\n')
        print('Start Training Epoch', epoch, ':', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        model.train()

        if args.fp16 and args.n_gpu > 1:
            train_sampler.set_epoch(epoch)

        train_loss = 0
        lr = optimizer.param_groups[0]['lr']
        for step, (qids, qidxs, dids, didxs, sidxs, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            with autocast():
                output = model(qidxs, didxs, sidxs)

                # labels: two elements. every element's shape = tensor([bz])
                # so: two (bz, 1) -> (bz, 2)
                labels = torch.cat([x[:, None] for x in labels], dim=-1)
                labels = torch.as_tensor(
                    labels, dtype=torch.float, device=args.device)

                loss = criterion(output, labels.to(args.device))

                # lmd = .01
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

            # if args.local_rank in [-1, 0]:
            #     writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            #     train_loss += loss.item()
            #     writer.flush()
            train_loss += loss.item()

            args.global_step += 1

        # val_loss = evaluate(args, train_loader, model, criterion, 'train')

        # if args.local_rank in [-1, 0]:
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

    # if args.local_rank in [-1, 0]:
    print('Training Time:', int(time.time() - start))
    print('End time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
