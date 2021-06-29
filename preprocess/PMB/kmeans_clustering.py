import torch
import numpy as np
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.cluster import KMeans


if __name__ == '__main__':
    parser = ArgumentParser('Kmeans for PMB')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--clustering_data_file', type=str)
    parser.add_argument('--patterns_num', type=int, default=20)
    args = parser.parse_args()

    fn_embeddings = torch.load(
        '../ROT/data/{}/FN_{}_embeddings_static.pt'.format(args.dataset, args.pretrained_model))
    dn_embeddings = torch.load(
        '../ROT/data/{}/DN_{}_embeddings_static.pt'.format(args.dataset, args.pretrained_model))
    emsize = fn_embeddings.shape[-1]

    init_data = pickle.load(open(args.clustering_data_file, 'rb'))
    samples = sum([len(a) for b in init_data.values() for a in b.values()])
    print(len(init_data), sum([len(v) for v in init_data.values()]), samples)

    clusterIdx2dataIdx = dict()
    X = np.zeros((samples, emsize))
    cidx = 0

    for qidx in tqdm(init_data.keys()):
        for didx, sents in init_data[qidx].items():
            for sidx in sents:
                # residual embeddings
                X[cidx] = dn_embeddings[didx][sidx] - fn_embeddings[qidx]

                clusterIdx2dataIdx[cidx] = (qidx, didx, sidx)
                cidx += 1

    np.save('./data/{}/clustering_X.npy'.format(args.dataset), X)
    pickle.dump(clusterIdx2dataIdx, open(
        './data/{}/clustering_X_dataIdx.pkl'.format(args.dataset), 'wb'))

    kmeans = KMeans(n_clusters=args.patterns_num, random_state=0, verbose=1)
    print(kmeans)
    kmeans.fit(X)

    pickle.dump(kmeans, open(
        './data/{}/kmeans.pkl'.format(args.dataset), 'wb'))
    np.save('./data/{}/kmeans_cluster_centers.npy'.format(args.dataset),
            kmeans.cluster_centers_)
