import torch
import torch.nn as nn
from transformers import BertModel
import pickle
import os
import numpy as np
from tqdm import tqdm

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
        print('BERT pretrained model:'.format(args.pretrained_model))
        print('ROT model file:', args.rouge_bert_model_file)
        print('Memory initialization file:', args.memory_init_file)
        print('Claim-Sentence file:', args.claim_sentence_scores_file)
        print('*'*20)

        # Loading tokens
        FN_tokens_file = '../tokenize/data/{}/FN_{}.pkl'.format(
            args.dataset, args.pretrained_model)
        DN_tokens_file = '../tokenize/data/{}/DN_{}.pkl'.format(
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
                args.rouge_bert_model_file, map_location='cuda')

            # update bert's weights
            updated_weights_layers = []
            updated_model_dict = dict()
            for name in self.bert.state_dict():
                if 'embeddings' in name or 'encoder.layer.0' in name:
                    updated_model_dict[name] = rouge_bert_dict[name]
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

        # Claim-Sentence Scores
        with open(self.args.claim_sentence_distance_file, 'rb') as f:
            self.claim_sentence_distance_dict = pickle.load(f)

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
            train_dataset = DatasetLoader(self.args.topk, 'train.line', nrows=20 * self.args.batch_size,
                                          dataset=self.args.dataset)
            val_dataset = DatasetLoader(
                self.args.topk, 'val', nrows=2, dataset=self.args.dataset)
            test_dataset = DatasetLoader(
                self.args.topk, 'test', nrows=2, dataset=self.args.dataset)
        else:
            train_dataset = DatasetLoader(
                self.args.topk, 'train.line', dataset=self.args.dataset)
            val_dataset = DatasetLoader(
                self.args.topk, 'val', dataset=self.args.dataset)
            test_dataset = DatasetLoader(
                self.args.topk, 'test', dataset=self.args.dataset)

        train_loader = DatasetLoader(train_dataset, batch_size=1)
        val_loader = DatasetLoader(val_dataset, batch_size=1)
        test_loader = DatasetLoader(test_dataset, batch_size=1)

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

        print('\nUpdating the results of memory selector......\n')
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
                self.lambdaP * pattern_sentence_scores[sidx]
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
        # important_sentences_vidxs = []
        important_centers = None

        if self.AGGREGATE_MODE in [4, 5, 6]:
            # (bz, 768, 5)
            important_centers = self._tensorize(
                torch.zeros((len(qidx), len(self.memory[0]), selected_num)), type=torch.float)

        for i, fn_index in enumerate(qidx):
            fn_index = fn_index.item()
            dn_index = didx[i].item()

            selected_sentences_weights = []

            if self.METRIC_MODE in [0, 1]:
                # (, 768)
                fn = self.fn_arr[fn_index]
                # (#sent, 768)
                dn = self.dn_sent_arr[list(self.dn_sent_index[dn_index].values())]

                # (1, #sent)
                sim = pytorch_cos_sim(fn, dn)
                selected_sentences = [self.d_tokens_sentences[dn_index][r]
                                      for r in sim.argsort(descending=True)[0][:selected_num]]

            elif self.METRIC_MODE == 2:
                fn = self.fn_tokens[fn_index]
                dns = self.dn_sent_tokens[dn_index]

                # (hypo=dns, ref=fn)
                scores = self.rouge_model.get_scores([' '.join(map(str, dn)) for dn in dns],
                                                     [' '.join(map(str, fn)) for _ in dns])
                scores = torch.tensor([s['rouge-2']['f'] for s in scores])
                selected_sentences = [self.d_tokens_sentences[dn_index][r]
                                      for r in scores.argsort(descending=True)[:selected_num]]

            elif self.METRIC_MODE in [3, 5]:
                selected_sentences = [self.d_tokens_sentences[dn_index][r]
                                      for r in self.rouge_sim[fn_index][dn_index][:selected_num]]

            elif self.METRIC_MODE == 4:
                try:
                    rouge_scores_fn = self.rouge_bert_sim[fn_index]
                    rouge_scores_fn_dn = rouge_scores_fn[dn_index]
                    selected_sentences = [self.d_tokens_sentences[dn_index][r]
                                          for r in rouge_scores_fn_dn[:selected_num]]
                except:
                    selected_sentences = [self.d_tokens[dn_index]]

            elif self.METRIC_MODE == 6:
                if self.MEMORY_INIT_MODE != -1 and self.MEMORY_UPDATED_MODE == 1:
                    # results: (ranked_sidxs, ranked_sidxs_vidxs, ranked_scores)
                    results = self.get_ranked_sentences(
                        qidx=fn_index, didx=dn_index,
                        indicating_pairs=self.get_indicating_pairs(qidx=fn_index, didx=dn_index))

                    if fn_index not in self.tmp_selector.keys():
                        self.tmp_selector[fn_index] = {dn_index: results}
                    else:
                        self.tmp_selector[fn_index][dn_index] = results

                results = self.tmp_selector[fn_index][dn_index]

                selected_sidxs = results[0][:selected_num]
                selected_sidxs_vidxs = results[1][:selected_num]
                selected_sentences = [self.d_tokens_sentences[dn_index][r]
                                      for r in selected_sidxs]

                if self.AGGREGATE_MODE not in [0, 2]:
                    # print('【results[2][:5]】', results[2][:selected_num])
                    # unknown bug check
                    if len(results[2]) >= 1 and torch.isnan(results[2][0]):
                        indicating_pairs = self.get_indicating_pairs(qidx=fn_index, didx=dn_index)

                        Q = self.memory_vec_dict['Q'][fn_index]
                        Ss = self.memory_vec_dict['S'][dn_index]

                        relevance_ds = [self.relevance_distance_dict[fn_index][dn_index][i] for i in range(len(Ss))]
                        indicating_ds = [torch.norm((S - Q) - self.memory, p=2, dim=1) for S in Ss]

                        print('*' * 20)
                        print('results:\t{}\n'.format(results))
                        print('indicating_pairs:\t{}\n'.format(indicating_pairs))
                        print('relevance_distances:\t{}\n'.format(relevance_ds))
                        print('indicating_distances:\t{}\n'.format(indicating_ds))
                        print('*' * 20)

                        print('!' * 30)
                        exit()

                    mode = 'avg' if self.AGGREGATE_MODE == 6 else 'weighted'
                    selected_sentences_weights = get_normalized_weights(
                        results[2][:selected_num], numbers_type='score', mode=mode)

            # 将所有的样本，挑出的句子都补成5个
            selected_sentences += [[] for _ in range(selected_num - len(selected_sentences))]
            selected_sentences_weights += [.0 for _ in range(selected_num - len(selected_sentences_weights))]
            selected_sidxs_vidxs += [0 for _ in range(selected_num - len(selected_sentences_weights))]

            important_sentences.append(selected_sentences)
            important_sentences_weights.append(selected_sentences_weights)
            # important_sentences_vidxs.append(selected_sidxs_vidxs)

            if important_centers is not None:
                for j, vidx in enumerate(selected_sidxs_vidxs):
                    important_centers[i, :, j] = self.memory[vidx]

        # return important_sentences, important_sentences_weights, important_sentences_vidxs
        return important_sentences, important_sentences_weights, important_centers

    def forward(self, qidx, didx):
        queries = [self.q_tokens[q][:self.query_maxlen] for q in qidx]
        important_sentences, important_sentences_weights, important_centers = self.get_important_sentences(qidx,
                                                                                                           didx)
        docs_sentences = []
        for sentences in important_sentences:
            docs_sentences.append([sent[:self.doc_maxlen] for sent in sentences])

        bert_pools = []
        bert_Q = []
        bert_S = []

        for s in range(self.args.selected_sentences):
            # q_masks, s_masks: bz大小的list，其中每个元素是shape为(1, maxlen)的tensor
            input_ids, attention_mask, token_type_ids, q_masks, s_masks = zip(
                *[self._encode(queries[i], docs_sentences[i][s]) for i in range(len(queries))])
            input_ids, attention_mask, token_type_ids = self._tensorize(input_ids), self._tensorize(
                attention_mask), self._tensorize(token_type_ids)

            # sequence_output: (bz, maxlen, 768), pooled_output: (bz, 768)
            sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)
            bert_pools.append(pooled_output)

            # print()
            # print('pooled_output: ', pooled_output.shape)
            # print('sequence_output: ', sequence_output.shape)
            # print('q_masks: ', type(q_masks), len(q_masks))

            # (bz, maxlen)
            Q_mask = self._tensorize(torch.cat(q_masks, dim=0))
            S_mask = self._tensorize(torch.cat(s_masks, dim=0))
            # print('Q_mask:', Q_mask.shape)
            # print()

            # (bz, maxlen, 1) * (bz, maxlen, 768) / (bz, 1, 1) -> (bz, maxlen, 768)
            Q = Q_mask[:, :, None] * sequence_output / (torch.sum(Q_mask, dim=1)[:, None, None] + ZERO)
            S = S_mask[:, :, None] * sequence_output / (torch.sum(S_mask, dim=1)[:, None, None] + ZERO)

            # (bz, maxlen, 768) -> (bz, 768)
            bert_Q.append(torch.sum(Q, dim=1))
            bert_S.append(torch.sum(S, dim=1))

        # 先把每个元素变为(bz, 768, 1)，之后再拼接为(bz, 768, 5)
        bert_Q = torch.cat([Q[:, :, None] for Q in bert_Q], dim=2)
        bert_S = torch.cat([S[:, :, None] for S in bert_S], dim=2)

        concat = None
        aggregate_pooling = None

        if self.AGGREGATE_MODE == 0:
            bert_pools = torch.cat([pooled[:, :, None] for pooled in bert_pools], dim=2)
            # avg pooling
            aggregate_pooling = torch.mean(bert_pools, dim=2)

        else:
            # weights: (bz, 5) -> (bz, 1, 5)
            weights = self._tensorize(important_sentences_weights, type=torch.float)
            weights = weights[:, None, :]

            if self.AGGREGATE_MODE == 1:
                bert_pools = torch.cat([pooled[:, :, None] for pooled in bert_pools], dim=2)
                # weighted pooling
                aggregate_pooling = torch.sum(weights * bert_pools, dim=2)

            elif self.AGGREGATE_MODE == 2:
                # (bz, 768, 5) and (bz, 768, 5) -> (bz, 768*2, 5)
                concat = torch.cat([bert_Q, bert_S], dim=1)
                # [Q, S], avg pooling
                aggregate_pooling = torch.mean(concat, dim=2)

            elif self.AGGREGATE_MODE == 3:
                # concat: (bz, 768*2, 5), weights: (bz, 1, 5)
                concat = torch.cat([bert_Q, bert_S], dim=1)
                # [Q, S], weighted pooling
                aggregate_pooling = torch.sum(weights * concat, dim=2)

            elif self.AGGREGATE_MODE in [4, 6]:
                # (bz, 768*3, 5)
                concat = torch.cat([bert_Q, bert_S, important_centers], dim=1)
                # [Q, S, V], weighted pooling
                aggregate_pooling = torch.sum(weights * concat, dim=2)

            elif self.AGGREGATE_MODE == 5:
                # transfer: (bz, 768, 5) -> (bz, 5, 768)
                Q = bert_Q.permute(0, 2, 1)
                S = bert_S.permute(0, 2, 1)
                V = important_centers.permute(0, 2, 1)

                # (bz, 5, 768, 1) x (bz, 5, 1, 768) => (bz, 5, 768, 768)
                QS = torch.matmul(Q[:, :, :, None], S[:, :, None, :])
                # flatten (bz, 5, 768, 768) -> (bz, 5, 768 * 768) -> softmax (on dim -1) -> reshape
                QS = self.softmax(QS.flatten(start_dim=-2)).reshape(QS.shape)
                # (bz, 5, 768, 768) x (bz, 5, 768, 1) => (bz, 5, 768, 1) -> (bz, 5, 768)
                S_Q = torch.matmul(QS, S[:, :, :, None]).squeeze()

                VS = torch.matmul(V[:, :, :, None], S[:, :, None, :])
                VS = self.softmax(VS.flatten(start_dim=-2)).reshape(VS.shape)
                S_V = torch.matmul(VS, S[:, :, :, None]).squeeze()

                # 5个(bz, 5, 768) -> (bz, 5, 768*5) -> (bz, 768*5, 5)
                concat = torch.cat([Q, S, V, S_Q, S_V], dim=2).permute(0, 2, 1)

                # [Q, S, V, S_Q, S_V], weighted pooling
                aggregate_pooling = torch.sum(weights * concat, dim=2)

        # (bz, DIM) -> ... -> (bz, 2)
        mlp = self.fcs[0](aggregate_pooling)
        for fc in self.fcs[1:]:
            mlp = fc(mlp)

        mlp_drop_out = self.dropout(mlp)
        out = F.gelu(mlp_drop_out)

        # if torch.any(torch.isnan(out)):
        #     for error_i, o in enumerate(out):
        #         if torch.any(torch.isnan(o)):
        #             print('-' * 20)
        #             print('out: \t{}\n'.format(o))
        #             # print('aggregate_pooling:\t{}\n'.format(aggregate_pooling))
        #             print('concat:\t{}\n'.format(concat[error_i]))
        #             # print('bert_Q:\t{}\n'.format(bert_Q))
        #             # print('bert_S:\t{}\n'.format(bert_S))
        #             print('Q:\t{}\n'.format(Q[error_i]))
        #             print('S:\t{}\n'.format(S[error_i]))
        #             print('V:\t{}\n'.format(V[error_i]))
        #             print('QS:\t{}\n'.format(QS[error_i]))
        #             print('S_Q:\t{}\n'.format(S_Q[error_i]))
        #             print('VS:\t{}\n'.format(VS[error_i]))
        #             print('S_V:\t{}\n'.format(S_V[error_i]))
        #             print('-' * 20)

        #             with open(os.path.join(self.memory_save_dir, 'error_output.pkl'), 'wb') as f:
        #                 results = [bert_Q, bert_S, important_centers, Q, S, V, QS, S_Q, VS, S_V]
        #                 results += [weights, aggregate_pooling, mlp, mlp_drop_out, out]

        #                 results = [r.detach().cpu().numpy() for r in results]
        #                 pickle.dump(results, f)

        #             exit()

        return out


    def _encode(self, q, d):
        q = q[:self.query_maxlen]
        d = d[:self.doc_maxlen]

        padding_length = self.maxlen - (len(q) + len(d) + 3)

        if len(d) != 0:
            attention_mask = [1] * (len(q) + len(d) + 3) + [0] * padding_length
        else:
            attention_mask = [1] * (len(q) + 1) + [0] * \
                (self.maxlen - (len(q) + 1))

        input_ids = [101] + q + [102] + d + [102] + [103] * padding_length
        token_type_ids = [0] * (len(q) + 2) + [1] * (self.maxlen - len(q) - 2)

        return input_ids, attention_mask, token_type_ids

    def _tensorize(self, l, type=torch.long):
        return torch.tensor(l, dtype=type, device=self.args.device)


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
