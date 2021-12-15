import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel
import pickle
import os
import numpy as np
from tqdm import tqdm
import json
from functools import reduce

from DatasetLoader import DatasetLoader

ZERO = 1e-8


class MTM(nn.Module):
    def __init__(self, args):
        super(MTM, self).__init__()
        self.args = args

        # Config
        print('*'*20)
        print('Selecting {} sentences in an article...'.format(
            args.selected_sentences))
        print('BERT pretrained model:', args.pretrained_model)
        print('ROT model file:', args.rouge_bert_model_file)
        print('Memory initialization file:', args.memory_init_file)
        print('Claim-Sentence file:', args.claim_sentence_distance_file)
        print('Pattern-Sentence init file:',
              args.pattern_sentence_distance_init_file)
        print('*'*20)

        # Loading tokens
        FN_tokens_file = '../preprocess/tokenize/data/{}/FN_{}.pkl'.format(
            args.dataset, args.pretrained_model)
        DN_tokens_file = '../preprocess/tokenize/data/{}/DN_{}.pkl'.format(
            args.dataset, args.pretrained_model)
        self.q_tokens = pickle.load(open(FN_tokens_file, 'rb'))
        self.d_tokens_sentences = pickle.load(open(DN_tokens_file, 'rb'))

        # Initialization
        self.init()

        # Multi-layer Transformer
        self.bert = BertModel.from_pretrained(
            args.pretrained_model, return_dict=False)

        if args.rouge_bert_model_file != '':
            # Loading ROT
            rouge_bert_dict = torch.load(
                args.rouge_bert_model_file, map_location='cuda')['state_dict']

            # print(rouge_bert_dict.keys())

            # update bert's weights
            updated_weights_layers = []
            updated_model_dict = dict()
            for name in self.bert.state_dict():
                if 'embeddings' in name or 'encoder.layer.0' in name:
                    updated_model_dict[name] = rouge_bert_dict['model.{}'.format(
                        name)]
                    updated_weights_layers.append(name)
                else:
                    updated_model_dict[name] = self.bert.state_dict()[name]
            self.bert.load_state_dict(updated_model_dict)

            print('Loading from ROT, {}/{} layers: \n{}'.format(
                len(updated_weights_layers), len(self.bert.state_dict()),
                updated_weights_layers))

        for name, param in self.bert.named_parameters():
            # finetune the pooler layer
            if name.startswith("pooler"):
                if 'bias' in name:
                    param.data.zero_()
                elif 'weight' in name:
                    param.data.normal_(
                        mean=0.0, std=self.bert.config.initializer_range)
                param.requires_grad = True

            # finetune the last encoder layer
            elif name.startswith('encoder.layer.11'):
                param.requires_grad = True

            # the embedding layer and the first encoder layer
            elif name.startswith('encoder.layer.0') or name.startswith('embeddings'):
                param.requires_grad = args.finetune_front_layers

            # the other layers (intermediate layers)
            else:
                param.requires_grad = args.finetune_inter_layers

        fixed_layers = []
        for name, param in self.bert.named_parameters():
            if not param.requires_grad:
                fixed_layers.append(name)

        print("Fixed, {}/{} layers:\n{}".format(
            len(fixed_layers), len(self.bert.state_dict()), fixed_layers))

        self.maxlen = args.bert_max_length
        self.query_maxlen = args.query_max_length
        self.doc_maxlen = self.maxlen - args.query_max_length - 3

        self.fcs = []

        last_output = 3 * args.emsize
        for i in range(args.bert_mlp_layers - 1):
            curr_output = 1024 if last_output == args.emsize else int(
                last_output / 2)
            fc = nn.Linear(last_output, curr_output)
            last_output = curr_output
            self.fcs.append(fc)
        self.fcs.append(nn.Linear(last_output, 2, bias=False))
        self.fcs = nn.ModuleList(self.fcs)

        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)

    def init(self):
        # Vector Dictionary
        fn_embeddings = torch.load(
            '../preprocess/ROT/data/{}/FN_{}_embeddings_static.pt'.format(self.args.dataset, self.args.pretrained_model))
        dn_embeddings = torch.load(
            '../preprocess/ROT/data/{}/DN_{}_embeddings_static.pt'.format(self.args.dataset, self.args.pretrained_model))

        fn_embeddings = [self._tensorize(
            arr, type=torch.float) for arr in fn_embeddings]
        dn_embeddings = [[self._tensorize(
            arr, type=torch.float) for arr in sentences] for sentences in dn_embeddings]

        self.memory_vec_dict = {'Q': fn_embeddings, 'S': dn_embeddings}

        # Claim-Sentence distance dict
        if os.path.exists(self.args.claim_sentence_distance_file):
            with open(self.args.claim_sentence_distance_file, 'rb') as f:
                self.claim_sentence_distance_dict = pickle.load(f)
        else:
            print('\n{}\n'.format('*'*20))
            print('Init claim-sentence distance dict ...\n')

            train_dataset = DatasetLoader(
                'train.line', dataset=self.args.dataset)
            val_dataset = DatasetLoader('val', dataset=self.args.dataset)
            test_dataset = DatasetLoader('test', dataset=self.args.dataset)

            train_loader = DataLoader(train_dataset, batch_size=1)
            val_loader = DataLoader(val_dataset, batch_size=1)
            test_loader = DataLoader(test_dataset, batch_size=1)

            def add_elem_to_dict(qidx, didx, updated_dict):
                Q = fn_embeddings[qidx]
                Ss = dn_embeddings[didx]

                # dists = [pytorch_euclidean_distance(Q, S) for S in Ss]

                # (#sents, 768)
                Ss = torch.cat([S[None, :] for S in Ss], dim=0)
                dists = torch.norm(Ss - Q, p=2, dim=1)
                dists = dists.tolist()

                if qidx not in updated_dict.keys():
                    updated_dict[qidx] = {didx: dists}
                else:
                    updated_dict[qidx][didx] = dists

            claim_sentence_distance_dict = dict()

            for _, qidx, _, didx, _ in tqdm(train_loader):
                add_elem_to_dict(qidx.item(), didx.item(),
                                 claim_sentence_distance_dict)

            for loader in [val_loader, test_loader]:
                for _, qidx, _, didxs, _ in tqdm(loader):
                    for didx in didxs:
                        add_elem_to_dict(qidx.item(), didx.item(),
                                         claim_sentence_distance_dict)

            self.claim_sentence_distance_dict = claim_sentence_distance_dict
            with open(self.args.claim_sentence_distance_file, 'wb') as f:
                pickle.dump(self.claim_sentence_distance_dict, f)

            print('Done.')
            print('\n{}\n'.format('*'*20))

        # Memory
        self.memory = np.load(self.args.memory_init_file)
        self.memory = self._tensorize(self.memory, type=torch.float)

        self.memory_save_dir = os.path.join(self.args.save, 'memory')
        if not os.path.exists(self.memory_save_dir):
            os.makedirs(self.memory_save_dir)

        # Key Sentence Selector
        if not os.path.exists(self.args.pattern_sentence_distance_init_file):
            self.update_key_sentence_selector(
                self.args.pattern_sentence_distance_init_file)
        else:
            tmp_pattern_pairs = pickle.load(
                open(self.args.pattern_sentence_distance_init_file, 'rb'))
            self.tmp_selector = self.gen_key_sentence_selector(
                tmp_pattern_pairs)

    def get_pattern_pairs(self, qidx, didx):
        Q = self.memory_vec_dict['Q'][qidx]

        # [(pattern_d, vidx), ...]
        pattern_pairs = [self.get_pattern_sentence_distance(
            Q, S) for S in self.memory_vec_dict['S'][didx]]
        return pattern_pairs

    def get_pattern_sentence_distance(self, Q, S):
        distances = torch.norm((S - Q) - self.memory, p=2, dim=1)
        center_idx = torch.argmin(distances).item()
        pattern_distance = distances[center_idx].item()

        return pattern_distance, center_idx

    def update_key_sentence_selector(self, save_file):
        print('\n{}\n'.format('*'*20))
        print('Updating the results of Key Sentence Selector...\n')

        if self.args.debug:
            train_dataset = DatasetLoader(
                'train.line', nrows=20 * self.args.batch_size, dataset=self.args.dataset)
            val_dataset = DatasetLoader(
                'val', nrows=20, dataset=self.args.dataset)
            test_dataset = DatasetLoader(
                'test', nrows=20, dataset=self.args.dataset)
        else:
            train_dataset = DatasetLoader(
                'train.line', dataset=self.args.dataset)
            val_dataset = DatasetLoader('val', dataset=self.args.dataset)
            test_dataset = DatasetLoader('test', dataset=self.args.dataset)

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        tmp_pattern_pairs = dict()

        def add_elem_to_dict(qidx, didx):
            pattern_pairs = self.get_pattern_pairs(qidx, didx)
            if qidx not in tmp_pattern_pairs.keys():
                tmp_pattern_pairs[qidx] = {didx: pattern_pairs}
            else:
                tmp_pattern_pairs[qidx][didx] = pattern_pairs

        for _, qidx, _, didx, _ in tqdm(train_loader):
            add_elem_to_dict(qidx.item(), didx.item())

        for loader in [val_loader, test_loader]:
            for _, qidx, _, didxs, _ in tqdm(loader):
                for didx in didxs:
                    add_elem_to_dict(qidx.item(), didx.item())

        pickle.dump(tmp_pattern_pairs, open(save_file, 'wb'))
        print('\nDone, #samples = {}, #pairs = {}, saving in {}'.format(
            len(tmp_pattern_pairs), sum([len(v) for v in tmp_pattern_pairs.values()]), save_file))

        print('\nUpdating the results of key sentence selector......\n')
        self.tmp_selector = self.gen_key_sentence_selector(tmp_pattern_pairs)
        print('Done.')
        print('\n{}\n'.format('*'*20))

    def get_ranked_sentences(self, qidx, didx, pattern_paris):
        """
        :param pattern_paris: [(pattern_d, vidx), ...]
        :return:
            ranked_sidxs: the index of sentences
            ranked_sidxs_vidxs: the index of Memory w.r.t. ranked_sidxs
            ranked_scores: the scores of sentences w.r.t. ranked_sidxs
        """

        # distances -> scores -> scaled scores
        claim_sentence_distances = self.claim_sentence_distance_dict[qidx][didx]
        pattern_sentence_distances = [pair[0] for pair in pattern_paris]

        claim_sentence_scores = get_scaled_scores(claim_sentence_distances)
        pattern_sentence_scores = get_scaled_scores(pattern_sentence_distances)

        results = []
        for sidx, (_, vidx) in enumerate(pattern_paris):
            score = self.args.lambdaQ * \
                claim_sentence_scores[sidx] + \
                self.args.lambdaP * pattern_sentence_scores[sidx]
            results.append((score, vidx))

        scores = torch.tensor([r[0] for r in results])
        ranked_sidxs = torch.argsort(scores, descending=True).tolist()
        ranked_sidxs_vidxs = [results[sidx][1] for sidx in ranked_sidxs]
        ranked_scores = [scores[sidx] for sidx in ranked_sidxs]
        return ranked_sidxs, ranked_sidxs_vidxs, ranked_scores

    def gen_key_sentence_selector(self, tmp_pattern_paris):
        selector = dict()
        for qidx in tmp_pattern_paris.keys():
            for didx, pattern_paris in tmp_pattern_paris[qidx].items():
                if qidx not in selector.keys():
                    selector[qidx] = {didx: self.get_ranked_sentences(
                        qidx, didx, pattern_paris)}
                else:
                    selector[qidx][didx] = self.get_ranked_sentences(
                        qidx, didx, pattern_paris)

        return selector

    def get_important_sentences(self, qidx, didx):
        selected_num = self.args.selected_sentences

        important_sentences = []
        important_sentences_weights = []
        # (bz, 768, selected_num)
        important_centers = self._tensorize(
            torch.zeros((len(qidx), len(self.memory[0]), selected_num)), type=torch.float)

        for i, fn_index in enumerate(qidx):
            fn_index = fn_index.item()
            dn_index = didx[i].item()

            selected_sentences_weights = []

            results = self.tmp_selector[fn_index][dn_index]

            selected_sidxs = results[0][:selected_num]
            selected_sidxs_vidxs = results[1][:selected_num]
            selected_sentences = [self.d_tokens_sentences[dn_index][r]
                                  for r in selected_sidxs]

            selected_sentences_weights = get_normalized_weights(
                results[2][:selected_num], numbers_type='score', mode='weighted')

            # Padding to the selected_num sentences
            selected_sentences += [[]
                                   for _ in range(selected_num - len(selected_sentences))]
            selected_sentences_weights += [.0 for _ in range(
                selected_num - len(selected_sentences_weights))]
            selected_sidxs_vidxs += [0 for _ in range(
                selected_num - len(selected_sentences_weights))]

            # Append
            important_sentences.append(selected_sentences)
            important_sentences_weights.append(selected_sentences_weights)
            for j, vidx in enumerate(selected_sidxs_vidxs):
                important_centers[i, :, j] = self.memory[vidx]

        return important_sentences, important_sentences_weights, important_centers

    def forward(self, qidx, didx):
        queries = [self.q_tokens[q][:self.query_maxlen] for q in qidx]
        important_sentences, important_sentences_weights, important_centers = self.get_important_sentences(qidx,
                                                                                                           didx)
        docs_sentences = []
        for sentences in important_sentences:
            docs_sentences.append([sent[:self.doc_maxlen]
                                   for sent in sentences])

        bert_pools = []
        bert_Q = []
        bert_S = []

        for s in range(self.args.selected_sentences):
            # q_masks, s_masks: batch_size's list. Each item is a (1, max_len) tensor
            input_ids, attention_mask, token_type_ids, q_masks, s_masks = zip(
                *[self._encode(queries[i], docs_sentences[i][s]) for i in range(len(queries))])
            input_ids, attention_mask, token_type_ids = self._tensorize(input_ids), self._tensorize(
                attention_mask), self._tensorize(token_type_ids)

            # sequence_output: (bz, maxlen, 768), pooled_output: (bz, 768)
            sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)
            bert_pools.append(pooled_output)

            # (bz, maxlen)
            Q_mask = self._tensorize(torch.cat(q_masks, dim=0))
            S_mask = self._tensorize(torch.cat(s_masks, dim=0))

            # (bz, maxlen, 1) * (bz, maxlen, 768) / (bz, 1, 1) -> (bz, maxlen, 768)
            Q = Q_mask[:, :, None] * sequence_output / \
                (torch.sum(Q_mask, dim=1)[:, None, None] + ZERO)
            S = S_mask[:, :, None] * sequence_output / \
                (torch.sum(S_mask, dim=1)[:, None, None] + ZERO)

            # (bz, maxlen, 768) -> (bz, 768)
            bert_Q.append(torch.sum(Q, dim=1))
            bert_S.append(torch.sum(S, dim=1))

        # (bz, 768, #sents)
        bert_Q = torch.cat([Q[:, :, None] for Q in bert_Q], dim=2)
        bert_S = torch.cat([S[:, :, None] for S in bert_S], dim=2)

        # weights: (bz, #sents) -> (bz, 1, #sents)
        weights = self._tensorize(
            important_sentences_weights, type=torch.float)
        weights = weights[:, None, :]

        # (bz, 768*3, #sents)
        concat = torch.cat([bert_Q, bert_S, important_centers], dim=1)
        # (bz, 768*3)
        aggregate_pooling = torch.sum(weights * concat, dim=2)

        # print('\naggregate_pooling: {}\n'.format(aggregate_pooling[:, :20]))

        # (bz, DIM) -> ... -> (bz, 2)
        mlp = aggregate_pooling
        for fc in self.fcs:
            mlp = fc(mlp)
        mlp = F.gelu(mlp)

        # mlp = F.gelu(self.fcs[0](aggregate_pooling))
        # for fc in self.fcs[1:]:
        #     mlp = F.gelu(fc(mlp))

        # print('\nmlp output: {}\n'.format(mlp))

        mlp_drop_out = self.dropout(mlp)
        out = F.gelu(mlp_drop_out)

        return out

    def _encode(self, q, d):
        q = q[:self.query_maxlen]
        d = d[:self.doc_maxlen]

        padding_length = self.maxlen - (len(q) + len(d) + 3)
        attention_mask = [1] * (len(q) + len(d) + 3) + [0] * padding_length
        input_ids = [101] + q + [102] + d + [102] + [103] * padding_length
        token_type_ids = [0] * (len(q) + 2) + [1] * (self.maxlen - len(q) - 2)

        q_mask = torch.zeros((1, self.maxlen))
        d_mask = torch.zeros((1, self.maxlen))
        q_mask[0][1:1 + len(q)] = 1
        d_mask[0][2 + len(q):2 + len(q) + len(d)] = 1

        return input_ids, attention_mask, token_type_ids, q_mask, d_mask

    def _tensorize(self, l, type=torch.long):
        return torch.as_tensor(l, dtype=type, device=self.args.device)

    def update_memory_after_epoch(self, predictions_file, epoch):
        with torch.no_grad():
            print('\n{}\n'.format('*'*20))
            print('Updating the PMB......\n')

            with open(predictions_file, 'r') as f:
                tmp_predictions = json.load(f)

            pos_samples_dict = dict()
            neg_samples_dict = dict()

            # Get pos/neg samples and weights for every memory_cluster
            for qidx, didx, y, y_hat in tmp_predictions:
                ranked_sidxs, ranked_sidxs_vidxs, _ = self.tmp_selector[qidx][didx]
                selected_sidxs = ranked_sidxs[:self.args.selected_sentences]
                selected_sidxs_vidxs = ranked_sidxs_vidxs[:self.args.selected_sentences]

                Q = self.memory_vec_dict['Q'][qidx]
                Ss = [self.memory_vec_dict['S'][didx][sidx]
                      for sidx in selected_sidxs]
                points = [torch.as_tensor(S - Q) for S in Ss]

                if (y_hat > 0.5) or (y_hat == 0.5 and y == 0):
                    samples_dict = pos_samples_dict
                else:
                    samples_dict = neg_samples_dict

                # when y_hat >= 0.5, the sample is in pos_samples:
                #       weight = y_hat - 0.5
                #       the more y_hat, the rightter prediction -> a more weight
                #
                # when y_hat < 0.5, the sample is in neg_samples:
                #       weight = 0.5 - y_hat
                #       the less y_hat, the wronger prediction -> a more weight

                weight = abs(y_hat - 0.5)

                for i, vidx in enumerate(selected_sidxs_vidxs):
                    if vidx not in samples_dict.keys():
                        samples_dict[vidx] = [(weight, points[i])]
                    else:
                        samples_dict[vidx].append((weight, points[i]))

            # Logging
            for vidx in range(len(self.memory)):
                pos_num = len(
                    pos_samples_dict[vidx]) if vidx in pos_samples_dict.keys() else 0
                neg_num = len(
                    neg_samples_dict[vidx]) if vidx in neg_samples_dict.keys() else 0
                print('Memory Cluster {}: #positive samples = {}, #negative samples = {}'.format(
                    vidx, pos_num, neg_num))

            # Updating every memory cluster

            # Step1: Calculate the center of samples
            for samples_dict in [pos_samples_dict, neg_samples_dict]:
                for vidx, samples in samples_dict.items():
                    weight_sum = sum([item[0] for item in samples]) + ZERO

                    samples = [item[0] / weight_sum * item[1]
                               for item in samples]
                    samples_dict[vidx] = (weight_sum, reduce(
                        lambda x, y: x + y, samples))

            # Step2: Draw closer / Push away the memory cluster
            for vidx, V in enumerate(self.memory):
                weight_pos = 0
                weight_neg = 0
                a = torch.zeros(V.shape)
                b = torch.zeros(V.shape)

                if vidx in pos_samples_dict.keys():
                    weight_pos, pos = pos_samples_dict[vidx]
                    a = pos - V
                if vidx in neg_samples_dict.keys():
                    weight_neg, neg = neg_samples_dict[vidx]
                    b = V - neg

                if weight_pos + weight_neg == 0:
                    continue

                alpha = weight_pos / (weight_pos + weight_neg)
                beta = weight_neg / (weight_pos + weight_neg)

                a = self._tensorize(a, type=torch.float)
                b = self._tensorize(b, type=torch.float)
                alpha = self._tensorize(alpha, type=torch.float)
                beta = self._tensorize(beta, type=torch.float)

                grad = alpha * a + beta * b
                grad /= torch.norm(grad)

                self.memory[vidx] += self.args.memory_updated_step * \
                    torch.norm(V) * grad

        # Saving
        save_file = 'memory_epoch{}.npy'.format(epoch)
        np.save(os.path.join(self.memory_save_dir, save_file),
                self.memory.detach().cpu().numpy())
        print('\nDone, saving in {}'.format(save_file))

        # Update key sentence selector
        pattern_sentence_distance_save_file = 'pattern_sentence_distance_epoch{}.pkl'.format(
            epoch)
        pattern_sentence_distance_save_file = os.path.join(
            self.memory_save_dir, pattern_sentence_distance_save_file)
        self.update_key_sentence_selector(pattern_sentence_distance_save_file)

        print('\n{}\n'.format('*'*20))


def get_scaled_scores(distances):
    """
    :param distances: the smaller distance, the more importance
    :return: scaled scores: (1) map to [0, 1] (2) the more score, the more importance
    """
    if len(distances) == 0:
        return distances

    # scores = [-1 * d for d in distances]
    # m, M = min(scores), max(scores)
    # scale = M - m if M - m != 0 else ZERO
    # return [(s - m) / scale for s in scores]

    m, M = min(distances), max(distances)
    return [1 - (d - m) / (M - m + ZERO) for d in distances]


def get_normalized_weights(sorted_numbers, numbers_type='score', mode='weighted'):
    """
    :param sorted_numbers
    :param numbers_type: ['score', 'distance']
    :param mode: ['weighted', 'avg']
    :return: weights: desending
    """

    assert numbers_type in ['score', 'distance']
    assert mode in ['weighted', 'avg']

    if len(sorted_numbers) == 1:
        return [1.0]

    if mode == 'avg':
        avg = 1.0 / len(sorted_numbers)
        weights = [avg for _ in sorted_numbers]
        return weights

    sorted_numbers = torch.as_tensor(sorted_numbers)

    # Normalize
    sorted_numbers /= torch.sum(sorted_numbers) + ZERO

    if numbers_type == 'distance':
        weights = (1 - sorted_numbers) / (len(sorted_numbers) - 1)
    else:
        weights = sorted_numbers

    return weights.tolist()


def pytorch_euclidean_distance(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return torch.dist(a, b)
