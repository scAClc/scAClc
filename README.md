# scAClc: A Multi-Objective Adaptive Clustering Framework for Single-Cell Transcriptomics via Contrastive and Resolution-Aware Representation Learning

## Abstract

Single-cell RNA sequencing (scRNA-seq) enables whole-transcriptomic profiling at the single-cell resolution, with the promise of constructing virtual cells that capture the full spectrum of cellular identities. Realizing this goal hinges on accurate clustering, which remains challenging due to data sparsity, high dimensionality, noise, and the need to specify cluster numbers a priori. We propose scAClc, a novel clustering framework featuring multi-objective optimization and adaptive resolution discovery, designed to address these limitations through three key innovations. First, a Hierarchical Gene Relevance Module integrates global gene variability with local neighborhood-specific signals to eliminate redundancy while retaining biologically informative features. Second, an Anchor-Centered Contrastive Learning Module adaptively selects representative anchors to guide embedding learning, promoting compact intra-cluster structure and clear inter-cluster separation. Third, based on the robust low-dimensional embeddings, we propose a Self-Adaptive Resolution Discovery Module to automatically infer the number of clusters by jointly modeling intra- and inter-cluster distances. Extensive experiments on fifteen real scRNA-seq datasets demonstrate that scAClc consistently outperforms five state-of-the-art methods across multiple evaluation metrics. Ablation studies further confirm the complementary contributions of each module. In addition, interpretability analysis effectively mitigates the “black box” nature of clustering models and sheds light on the biological mechanisms underlying cell clustering. The source code
is publicly available at https://github.com/scAClc/scAClc.

## Environment

* Anaconda
* Python 3.8
## Dependency

* Pytorch 2.4.1+cu121
* Torchvision 0.19.1+cu121
* Matplotlib 3.7.5
* numpy  1.24.3
* h5py  3.9.0
* pandas 1.3.5
* scanpy 1.9.8
* scipy 1.10.1
* scikit-learn 1.2.2
* networkx 2.6.3

## Installation

1. Download and install Anaconda.
   ```git clone https://github.com/scAClc/scAClc```

2. Create the prosperousplus environment.

   ```conda create -n scAClc python=3.8.10```

3. Activate the prosperousplus environment and install the dependencies.

   ```conda activate scAClc```

   ```pip install or conda install```


## Quick start

Load the data to be analyzed:

```python
import scanpy as sc

adata = sc.AnnData(data)
```

Perform data pre-processing:

```python
# Basic filtering
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)

adata.raw = adata.copy()

# Total-count normalize, logarithmize the data  
sc.pp.normalize_per_cell(adata)
adata.obs['scale_factor'] = adata.obs.n_counts / adata.obs.n_counts.median()

sc.pp.log1p(adata)
```

Run the scAClc method:

```python
from scaclc import run_scaclc
adata = run_scaclc(adata)
```

The output adata contains cluster labels in `adata.obs['scaclc_cluster']` and the cell embeddings in `adata.obsm['scaclc_emb']`. The embeddings can be used as input of other downstream analyses.

## Output:
NMI and ARI values ​​for clustering.

<ins>Please refer to `scAClc.ipynb` for a detailed description of scAClc's usage.<ins>

