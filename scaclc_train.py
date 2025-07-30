import scanpy as sc
import numpy as np
import torch
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import scAClcContrastive, centroid_merge, merge_compute
from .util import scDataset, ZINBLoss, ClusterLoss, ELOBkldLoss, clustering, calculate_metric, compute_mu,NTXentLoss
import time

import networkx as nx
from community import community_louvain
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA





def fast_clustering(data, k=15, seed=2023):

    nps = min(40, data.shape[0])
    if data.shape[0] >= nps:
        pca = PCA(n_components=nps, random_state=seed)
        X_pca = pca.fit_transform(data)
    else:

        X_pca = data

    from sklearn.neighbors import NearestNeighbors

    nn_res = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1).fit(X_pca)
    distances, indices = nn_res.kneighbors(X_pca)

    G = nx.Graph()
    for i in range(indices.shape[0]):

        for j in indices[i][1:]:
            G.add_edge(i, j)

    partition = community_louvain.best_partition(G, random_state=seed)
    return partition

def run_scaclc(adata: sc.AnnData,
              n_epochs_pre: int = 200,
              n_epochs: int = 500,
              n_epochs_finetune: int = 100,
              batch_size: int = 256,
              lr: float = 1e-4,
              resolution: float = 2,
              init_cluster=None,
              init_method='leiden',
              cl_type=None,
              save_pretrain: bool = False,
              saved_ckpt: str = None,
              pretrained_ckpt: str = None,
              return_all: bool = False,
              seed: int = 2023,
              n_top_hvg: int = 3000,
              n_top_rfg: int = 3000

              ):
    """
        Train scAClc.
        Parameters
        ----------
        adata
            AnnData object of scanpy package.
        n_epochs_pre
            Number of total epochs in pre-training.
        n_epochs
            Number of total epochs in training.
        batch_size
            Number of cells for training in one epoch.
        lr
            Learning rate for AdamOptimizer.
        resolution
            The resolution parameter of sc.tl.leiden or sc.tl.louvain for the initial clustering.
        init_cluster
            Initial cluster results. If provided, perform cluster splitting after pre-training.
        init_method
            Method used for cluster initialization. Default is 'leiden', optionally input 'leiden', 'louvain' or 'kmeans'.
        save_pretrain
            If True, save the pre-trained model.
        saved_ckpt
            File name of pre-trained model to be saved, only used when save_pretrain is True.
        pretrained_ckpt
            File name of saved pre-trained model. If provided, load the saved pre-trained model without performing
            pre-training step.
        cl_type
            Cell type information. If provided, calculate ARI and NMI after clustering.
        return_all
            If True, print and return all temporary results.

        Returns
        -------
        adata
            AnnData object of scanpy package. Embedding and clustering result will be stored in adata.obsm['scaclc_emb']
            and adata.obs['scaclc_cluster']
        nmi
            Final NMI. Will be returned if 'return_all' is True and cell type information is provided.
        ari
            Final ARI. Will be returned if 'return_all' is True and cell type information is provided.
        K
            Final number of clusters. Will be returned if 'return_all' is True.
        pred_all
            All temporary clustering results. Will be returned if 'return_all' is True.
        emb_all
            All temporary embedding. Will be returned if 'return_all' is True.
        run_time
            Time spent in training.
    """

    ####################   Assert several input variables  ########################
    # To print and store all temporary results
    if return_all:
        pred_all, emb_all = [], []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################   Prepare data for training   ####################

    # 1.  HVGs and RFGs
    print(">>> Feature selection (HVGs and RFGs) in progress...")

    nan_genes = np.isnan(adata.X).any(axis=0)
    if np.sum(nan_genes) > 0:
        print(f">>> Removing {np.sum(nan_genes)} genes with NaN values after scaling.")
        valid_genes = adata.var_names[~nan_genes]
        adata = adata[:, valid_genes].copy()
        if adata.raw is not None:
            adata.raw = adata.raw[:, valid_genes].copy()
        print(f">>> adata now has {adata.X.shape[1]} genes.")

    # 1.1 HVGs
    print(">>> Identification of HVGs is currently in progress...")

    adata_for_hvg = adata.copy()
    if not isinstance(adata_for_hvg.X, np.ndarray):
        adata_for_hvg.X = adata_for_hvg.X.toarray()


    try:
        sc.pp.highly_variable_genes(adata_for_hvg, n_top_genes=n_top_hvg, flavor='seurat')
        hvgs = list(adata_for_hvg[:, adata_for_hvg.var.highly_variable].var_names)
        print(f">>> Identified {len(hvgs)} Highly Variable Genes.")
    except Exception as e:
        print(f"Error during HVG calculation: {e}")
        print("Falling back to selecting top genes by dispersion directly.")
        hvgs = []

    # 1.2 RFGs
    print(">>> Identification of RFGs is currently in progress...")

    X_for_rfg = adata.X.copy()
    n_cells = X_for_rfg.shape[0]

    Init_clusters = fast_clustering(data=X_for_rfg, seed=seed)
    Init_clusters = [Init_clusters[node] for node in range(n_cells)]

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=seed, bootstrap=False)
    rf.fit(X_for_rfg, Init_clusters)
    gene_imp = rf.feature_importances_.copy()

    rfgs = list(adata.var_names[np.argsort(-gene_imp)[:n_top_rfg]])
    print(f">>> Identified {len(rfgs)} Random Forest based Genes.")


    sg = sorted(set(hvgs).union(set(rfgs)))
    print(f">>> Total selected genes: {len(sg)}")


    adata = adata[:, sg].copy()
    if adata.raw is not None:

        original_raw_var_names = adata.raw.var_names.tolist()

        filtered_raw_var_names_in_sg_order = [gene for gene in sg if gene in original_raw_var_names]


        sg_indices_in_raw = [original_raw_var_names.index(gene) for gene in filtered_raw_var_names_in_sg_order]


        raw_X_filtered = adata.raw.X[:, sg_indices_in_raw]
        var_for_raw = adata.var.copy()

        adata.raw = sc.AnnData(
            X=raw_X_filtered,
            var=var_for_raw,
            obs=adata.obs
        )

    print(f">>> adata filtered to {adata.X.shape[1]} genes.")
    if adata.raw is not None:
        print(f">>> adata.raw also filtered to {adata.raw.X.shape[1]} genes.")

    # Prepare data
    raw_mat, exp_mat = adata.raw.X, adata.X
    cell_type = adata.obs[cl_type] if cl_type is not None else None

    # Assume that 'scale_factor' has been calculated
    if 'scale_factor' not in adata.obs:
        scale_factor = np.ones((exp_mat.shape[0], 1), dtype=np.float32)
    else:
        scale_factor = adata.obs['scale_factor'].values.reshape(-1, 1)

    # Create dataset
    train_dataset = scDataset(raw_mat, exp_mat, scale_factor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    n_iters = len(train_loader)

    ####################   Set some parameters   #################
    # hyper-parameter
    kld_w2 = 0.01
    kld_w1 = 0.001
    z_dim = 32
    encode_layers = [512]
    decode_layers = [512]
    activation = 'relu'

    # parameter for training
    tol, clu_w, m_numbers = 0.05, 1, 0
    merge_flag = True



    contrastive_w2=0.1

    #######################   Prepare models & optimzers & loss  #######################
    input_dim = adata.X.shape[1]

    scaclc_model = scAClcContrastive(input_dim=input_dim, device=device, z_dim=z_dim, encode_layers=encode_layers,
                                   decode_layers=decode_layers, activation=activation).to(device)

    optimizer = optim.Adam(params=scaclc_model.parameters(), lr=lr)

    ZINB_Loss, KLD_Loss, Cluster_Loss = ZINBLoss(ridge_lambda=0), ELOBkldLoss(), ClusterLoss()
    contrastive_loss = NTXentLoss(temperature=0.5)
    start = time.time()

    ###########################   Pre-training   #########################
    if pretrained_ckpt:
        print('Pre-trained model provided, load checkpoint from file "{}".'.format(pretrained_ckpt))

        scaclc_model.load_state_dict(torch.load(pretrained_ckpt))

    else:
        print('Start pre-training! Total epochs is {}.'.format(n_epochs_pre))

        scaclc_model.pretrain = True

        # Start pre-training
        for epoch in tqdm(range(n_epochs_pre), unit='epoch', desc='Pre-training:'):
            avg_zinb, avg_kld, avg_loss = 0., 0., 0.

            for idx, raw, exp, sf in train_loader:
                raw, exp, sf = raw.to(device), exp.to(device), sf.to(device)
                if scaclc_model.pretrain:

                    z_mu, z_logvar, z = scaclc_model.Encoder(exp)
                    mu, disp, pi = scaclc_model.Decoder(z)
                else:
                    z_mu, z_logvar, mu, disp, pi = scaclc_model(exp)[:5]

                # VAE Loss
                zinb_loss = ZINB_Loss(x=raw, mean=mu, disp=disp, pi=pi, scale_factor=sf)
                kld_loss = KLD_Loss(z_mu, z_logvar)

                loss = zinb_loss + kld_w1 * kld_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record losses
                if return_all:
                    avg_zinb += zinb_loss.item() / n_iters
                    avg_kld += kld_loss.item() / n_iters
                    avg_loss += loss.item() / n_iters

            if return_all:
                print('Pre-training epoch [{}/{}]. Average ZINB loss:{:.4f}, kld loss:{:.4f}, total loss:{:.4f}'
                      .format(epoch + 1, n_epochs_pre, avg_zinb, avg_kld, avg_loss))

    # Finish pre-training
    scaclc_model.pretrain = False

    y_pred_last, mu, scaclc_emb = clustering(scaclc_model, exp_mat, init_method=init_method, resolution=resolution)
    n_clusters = len(np.unique(y_pred_last))


    adata.obs['scaclc_cluster'] = y_pred_last
    scaclc_model.mu = Parameter(torch.Tensor(n_clusters, scaclc_model.z_dim).to(device))

    y_pred, scace_emb, q, p = clustering(scaclc_model, exp_mat)


    for param in scaclc_model.encoder.parameters():
        param.requires_grad = False
    optimizer = optim.Adam([
        {'params': scaclc_model.mu},
        {'params': scaclc_model.decoder.parameters()}
    ], lr=lr)

    #fine-tuning
    print('Start fine - tuning! Total fine - tuning epochs is {}.'.format(n_epochs_finetune))
    for epoch in tqdm(range(n_epochs_finetune), unit='epoch', desc='Fine - tuning:'):


        y_pred, scace_emb, q, p = clustering(scaclc_model, exp_mat)  # 确保p是最新的目标分布


        avg_zinb, avg_kld, avg_clu, avg_contrastive, avg_loss = 0., 0., 0., 0., 0.
        for idx, raw, exp, sf in train_loader:
            raw, exp, sf = raw.to(device), exp.to(device), sf.to(device)

            z_mu, z_logvar, z, z0 = scaclc_model.Encoder(exp)
            mu, disp, pi = scaclc_model.Decoder(z)
            h = scaclc_model.contrastive_head(z0)



            # VAE Losses
            zinb_loss = ZINB_Loss(x=raw, mean=mu, disp=disp, pi=pi, scale_factor=sf)
            kld_loss = KLD_Loss(z_mu, z_logvar)



            # Contrastive Loss
            contrastive = contrastive_loss(h)

            # All losses
            loss = zinb_loss + kld_w2 * kld_loss +  contrastive_w2 * contrastive


            # Optimize VAE + DEC
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            if return_all:
                avg_zinb += zinb_loss.item() / n_iters
                avg_kld += kld_loss.item() / n_iters
                avg_contrastive += contrastive.item() / n_iters
                avg_loss += loss.item() / n_iters

        if return_all:
            print(
                'Train epoch [{}/{}]. ZINB loss:{:.4f}, kld loss:{:.4f}, contrastive loss:{:.4f}, total loss:{:.4f}'.format(
                    epoch + 1, n_epochs, avg_zinb, avg_kld, avg_contrastive, avg_loss))


    for param in scaclc_model.encoder.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(scaclc_model.parameters(), lr=lr)


    scaclc_model.pretrain = False

    print('Finish pre-training!')

    if save_pretrain:
        torch.save(scaclc_model.state_dict(), saved_ckpt)

    ###########################   Find initial clustering centers  #########################

    # Initial clustering
    if init_cluster is not None:
        print('Perform initial clustering through cluster split with provided cluster labels')
        y_pred_last, mu, scaclc_emb = clustering(scaclc_model, exp_mat, init_cluster=init_cluster)

    else:
        if init_method == 'kmeans':
            print('Perform initial clustering through K-means')
        else:
            print('Perform initial clustering through {} with resolution = {}'.format(init_method, resolution))
        y_pred_last, mu, scaclc_emb = clustering(scaclc_model, exp_mat, init_method=init_method, resolution=resolution)



    # Number of initial clusters
    n_clusters = len(np.unique(y_pred_last))
    print('Finish initial clustering! Number of initial clusters is {}'.format(n_clusters))

    # Initial parameter mu
    scaclc_model.mu = Parameter(torch.Tensor(n_clusters, scaclc_model.z_dim).to(device))
    optimizer = optim.Adam(params=scaclc_model.parameters(), lr=lr)
    scaclc_model.mu.data.copy_(torch.Tensor(mu))

    # Store initial tsne plot and clustering result
    if return_all:
        emb_all.append(scaclc_emb)
        pred_all.append(y_pred_last)

        # If there provide ground truth cell type information, calculate NMI and ARI
        if cl_type is not None:
            nmi, ari = calculate_metric(y_pred_last, cell_type)
            print('Initial Clustering: NMI= %.4f, ARI= %.4f' % (nmi, ari))

    ############################   Training   #########################

    print('Start training! Total epochs is {}.'.format(n_epochs))

    # Calculate q, p firstly
    y_pred, scaclc_emb, q, p = clustering(scaclc_model, exp_mat)

    # Start training
    for epoch in tqdm(range(n_epochs), unit='epoch', desc='Training:'):

        # Check stop & cluster merging criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / len(y_pred)
        y_pred_last = y_pred
        contrastive_w2 = max(0.12 * (1 - epoch / (2 * n_epochs)), 0.08)
        if epoch > 0 and delta_label < tol:
            if not merge_flag: print("Reach tolerance threshold. Stopping training."); break

        if epoch > 0 and (delta_label < tol or epoch % 20 == 0):

            print('Reach tolerance threshold. Perform cluster merging.')

            mu_prepare = scaclc_model.mu.cpu().detach().numpy()
            y_pred, Centroid, d_bar, intra_dis, d_ave = merge_compute(y_pred, mu_prepare, scaclc_emb)



            Final_Centroid_merge, Label_merge, n_clusters_t, pred_t = centroid_merge(scaclc_emb, Centroid, y_pred, d_bar,
                                                                                     intra_dis, d_ave)
            m_numbers += 1

            # n_clusters not change, stop merging clusters
            if m_numbers > 1 and n_clusters_t == n_clusters:
                merge_flag, tol = False, tol / 10.
                print('Stop merging clusters! Continue updating several rounds.')

            else:
                n_clusters = n_clusters_t
                y_pred = Label_merge

                mu = compute_mu(scaclc_emb, y_pred)

                scaclc_model.mu = Parameter(torch.Tensor(n_clusters, scaclc_model.z_dim).to(device))
                optimizer = optim.Adam(params=scaclc_model.parameters(), lr=lr)
                scaclc_model.mu.data.copy_(torch.Tensor(mu))

                q = scaclc_model.soft_assign(torch.tensor(scaclc_emb).to(device))
                p = scaclc_model.target_distribution(q)

            # Store tsne plot and clustering results of each cluster merging
            if return_all:
                emb_all.append(scaclc_emb)
                pred_all.append(pred_t)

        # Start training
        avg_zinb, avg_kld, avg_clu, avg_contrastive, avg_loss = 0., 0., 0., 0., 0.

        for idx, raw, exp, sf in train_loader:
            raw, exp, sf = raw.to(device), exp.to(device), sf.to(device)

            z_mu, z_logvar, z, z0 = scaclc_model.Encoder(exp)
            mu, disp, pi = scaclc_model.Decoder(z)
            h = scaclc_model.contrastive_head(z0)

            q = scaclc_model.soft_assign(z0)

            # VAE Losses
            zinb_loss = ZINB_Loss(x=raw, mean=mu, disp=disp, pi=pi, scale_factor=sf)
            kld_loss = KLD_Loss(z_mu, z_logvar)

            # DEC Loss
            clu_loss = Cluster_Loss(p[idx].detach(), q)

            # Contrastive Loss
            contrastive = contrastive_loss(h)

            # All losses
            loss = zinb_loss + kld_w2 * kld_loss + clu_w * clu_loss + contrastive_w2 * contrastive

            # Optimize VAE + DEC
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            if return_all:
                avg_zinb += zinb_loss.item() / n_iters
                avg_kld += kld_loss.item() / n_iters
                avg_clu += clu_loss.item() / n_iters
                avg_contrastive += contrastive.item() / n_iters
                avg_loss += loss.item() / n_iters

        if return_all:
            print(
                'Train epoch [{}/{}]. ZINB loss:{:.4f}, kld loss:{:.4f}, cluster loss:{:.4f}, contrastive loss:{:.4f}, total loss:{:.4f}'.format(
                    epoch + 1, n_epochs, avg_zinb, avg_kld, avg_clu, avg_contrastive, avg_loss))



        # Update the targe distribution p
        y_pred, scaclc_emb, q, p = clustering(scaclc_model, exp_mat)

        if cl_type is not None:
            nmi, ari = calculate_metric(y_pred, cell_type)
            print('Clustering   %d: NMI= %.4f, ARI= %.4f, Delta=%.4f' % (
                epoch + 1, nmi, ari, delta_label))

    end = time.time()
    run_time = end - start
    print(f'Total time: {end - start} seconds')

    ############################   Return results   #########################
    adata.obsm['scaclc_emb'] = scaclc_emb
    adata.obs['scaclc_cluster'] = y_pred

    K = len(np.unique(y_pred))

    if return_all:
        if cl_type is not None:
            return adata, nmi, ari, K, pred_all, emb_all, run_time
        return adata, K, pred_all, emb_all, run_time

    return adata