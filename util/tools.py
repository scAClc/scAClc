import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import KMeans

from ..model import centroid_split

def compute_mu(scaclc_emb, pred):
    mu = []
    for idx in np.unique(pred):
        mu.append(scaclc_emb[idx == pred, :].mean(axis=0))

    return np.array(mu)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = pd.Series(data=y_pred)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    D = int(D)
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def clustering(model, exp_mat, init_cluster=None, init_method=None, resolution=None):
    model.eval()
    scaclc_emb = model.EncodeAll(exp_mat)
    model.train()


    if init_method == 'kmeans':
        scaclc_emb = scaclc_emb.cpu().numpy()
        max_score = -1
        k_init = 0
        for k in range(15, 31):

            kmeans = KMeans(k, n_init=50)
            y_pred = kmeans.fit_predict(scaclc_emb)
            s_score = metrics.silhouette_score(scaclc_emb, y_pred)
            if s_score > max_score:
                max_score = s_score
                k_init = k

        kmeans = KMeans(k_init, n_init=50)
        y_pred = kmeans.fit_predict(scaclc_emb)
        mu = kmeans.cluster_centers_
        return y_pred, mu, scaclc_emb


    elif init_method == 'leiden':
        adata_l = sc.AnnData(scaclc_emb.cpu().numpy())
        sc.pp.neighbors(adata_l, n_neighbors=10)
        sc.tl.leiden(adata_l, resolution=resolution, random_state=0)
        y_pred = np.asarray(adata_l.obs['leiden'], dtype=int)
        mu = compute_mu(scaclc_emb.cpu().numpy(), y_pred)

        return y_pred, mu, scaclc_emb.cpu().numpy()


    elif init_method == 'louvain':
        adata_l = sc.AnnData(scaclc_emb.cpu().numpy())
        sc.pp.neighbors(adata_l, n_neighbors=10)
        sc.tl.louvain(adata_l, resolution=resolution, random_state=0)
        y_pred = np.asarray(adata_l.obs['louvain'], dtype=int)
        mu = compute_mu(scaclc_emb.cpu().numpy(), y_pred)

        return y_pred, mu, scaclc_emb.cpu().numpy()


    if init_cluster is not None:
        cluster_centers = compute_mu(scaclc_emb.cpu().numpy(), init_cluster)

        data_1 = np.concatenate([scaclc_emb.cpu().numpy(), np.array(init_cluster).reshape(-1, 1)], axis=1)
        mu, y_pred = centroid_split(scaclc_emb.cpu().numpy(), data_1, cluster_centers, np.array(init_cluster))

        return y_pred, mu, scaclc_emb.cpu().numpy()


    # Deep Embedded Clustering
    q = model.soft_assign(scaclc_emb)
    p = model.target_distribution(q)

    y_pred = torch.argmax(q, dim=1).cpu().numpy()

    return y_pred, scaclc_emb.cpu().numpy(), q, p



def calculate_metric(pred, label):
    # acc = np.round(cluster_acc(label, pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 5)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 5)

    return nmi, ari


