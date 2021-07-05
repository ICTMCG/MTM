import torch
import pickle
import os
from tqdm import tqdm

from RougeBert import RougeBert
from config import parser


def get_total_embeddings(total_tokens, type='static'):
    """
    total_tokens: eg: [[1,2,3,4,5], [1,2,3], [1,2,4,5], ...]
    """

    total_embeddings = torch.zeros((len(total_tokens), emsize))

    batch_size = args.batch_size
    batch_num = len(total_tokens) // batch_size + 1

    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            queries = total_tokens[batch_size*i:batch_size*(i+1)]
            sentences = [[] for _ in queries]

            input_ids, attention_mask, token_type_ids = zip(
                *[rouge_bert_model._encode(queries[i], sentences[i]) for i in range(len(queries))])
            # input_ids, attention_mask: (batch_size, seq_len)
            input_ids, attention_mask, token_type_ids = rouge_bert_model._tensorize(input_ids), rouge_bert_model._tensorize(
                attention_mask), rouge_bert_model._tensorize(token_type_ids)
            # mask [CLS]
            attention_mask[:, 0] = 0

            if type == 'dynamic':
                # === dynamic: context-wise ===

                # seq_out: (batch_size, seq_len, emsize), pooled: (batch_size, emsize)
                seq_out, pooled = rouge_bert_model.model(
                    input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                out = seq_out

            if type == 'static':
                # === static: embedding layer ===

                # emb_out: (batch_size, seq_len, emsize)
                emb_out = embedding_layer.word_embeddings(input_ids)
                out = emb_out

            else:
                print('Select \'static\' or \'dynamic\'!')
                exit()

            # embeddings: (batch_size, emsize)
            embeddings = torch.sum(
                out * attention_mask[:, :, None], dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            total_embeddings[batch_size*i:batch_size*(i+1)] = embeddings

    return total_embeddings


if __name__ == '__main__':
    parser.add_argument('--rouge_bert_model_file', type=str)
    parser.add_argument('--embeddings_type', type=str,
                        default='static', help='[static, dynamic]')
    args = parser.parse_args()

    rouge_bert_model = RougeBert(args)
    rouge_bert_dict = torch.load(
        args.rouge_bert_model_file, map_location='cuda')
    rouge_bert_model.load_state_dict(rouge_bert_dict['state_dict'])
    rouge_bert_model.eval()

    # Embedding Layer
    embedding_layer = list(list(rouge_bert_model.children())[0].children())[0]
    emsize = embedding_layer.word_embeddings.embedding_dim
    print('RougeBert: {},\n\n Embedding Layer:\n {}\n'.format(
        args.rouge_bert_model_file, embedding_layer))

    with open('../tokenize/data/{}/FN_{}.pkl'.format(args.dataset, args.pretrained_model), 'rb') as f:
        fn_tokens = pickle.load(f)
    with open('../tokenize/data/{}/DN_{}.pkl'.format(args.dataset, args.pretrained_model), 'rb') as f:
        dn_tokens = pickle.load(f)
    dn_unzip_tokens = [tokens for sents in dn_tokens for tokens in sents]

    print('\nClaims\' Embeddings...\n')
    fn_embeddings = get_total_embeddings(fn_tokens, type=args.embeddings_type)
    print('\nArticles\' Embeddings...\n')
    dn_unzip_embeddings = get_total_embeddings(dn_unzip_tokens, type=args.embeddings_type)

    dn_embeddings = []
    curr = 0
    for sents in tqdm(dn_tokens):
        dn_embeddings.append(dn_unzip_embeddings[curr:curr+len(sents)])
        curr += len(sents)

    torch.save(fn_embeddings,
               'data/{}/FN_{}_embeddings_{}.pt'.format(args.dataset, args.pretrained_model, args.embeddings_type))
    torch.save(dn_embeddings,
               'data/{}/DN_{}_embeddings_{}.pt'.format(args.dataset, args.pretrained_model, args.embeddings_type))
