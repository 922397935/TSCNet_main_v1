# TSCNet: A Transformer-Simplicial Convolutional Network for Functional Brain Network Classification


## Datasets
ABIDE
Download the ABIDE dataset from [here](https://drive.google.com/file/d/14UGsikYH_SQ-d_GvY2Um2oEHw3WNxDY3/view?usp=sharing).




## Installation

```bash
conda create --name bnt python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge wandb
pip install hydra-core --upgrade
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
pip install nilearn==0.12.1
```


## Dependencies

  - python=3.9
  - cudatoolkit=11.3
  - torchvision=0.13.1
  - pytorch=1.12.1
  - torchaudio=0.12.1
  - wandb=0.13.1
  - scikit-learn=1.1.1
  - pandas=1.4.3
  - hydra-core=1.2.0

## Usage

1. Change the *path* attribute in file *source/conf/dataset/ABIDE.yaml* to the path of your dataset.

2. Run the following command to train the model.

```bash
python -m source --multirun datasz=100p model=snt,bnt,fbnetgen,brainnetcnn,transformer dataset=ABIDE,ABCD repeat_time=5 preprocess=mixup
```

```
- **datasz**, default=100p, optional values: (10p, 20p, 30p, 40p, 50p, 60p, 70p, 80p, 90p, 100p). How much data to use for training. The
  value is a percentage of the total number of samples in the dataset. For example, 10p means 10% of the total number of samples in the training set.
  
- **model**, optional values: (SNT, mixed_model, braingb, braingnn_orig, brainnetcnn, mlp_graph, mlp_node, neurograph,).
  Notably, 'mixed_model' includes GCN, GAT, GIN, GraphSage, Brain Network Tranformer and the proposed Dual-pathway model,
  it needs to be used in combination with parameters 'model.has_nonaggr_module','model.has_aggr_module' and 'model.aggr_module'.
  
- **dataset**, optional values: (ABIDE, ADHD, Haxby)
  
- **repeat_time**, number of cross-validation runs, default=10
  
- **dataset.measure**, measures (labels) needed to be predicted. ABCD:pea_wiscv_trs (fluid intelligence), HCP:PMAT24_A_CR (fluid intelligence)
  ABIDE:Autism, PNC:sex

- **dataset.node_feature_type**, choice of node feature for graph deep learning models' input, optional values: (connection, learnable_time_series).
  'conncetion' denotes connection profile. 'learnable_time_series' denotes learnable node features from BOLD timeseries.

- **dataset.only_positive_corr**, retain only positive correlation as edges in the brain networks or keep both positive and negative conncetions.

- **dataset.sparse_ratio**, graph densities, retaining the top K% edges in the graphs. Value ranges from (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1).
  Notably, dataset.sparse_ratio=0 denotes no edges exist between ROIs, meaning node features are transformed independently without any aggregation.

- **model.has_nonaggr_module, model.has_aggr_module, model.aggr_module**. When training the GCN/GAT/GIN/GraphSage/BrainNetTF model,
  set model.has_nonaggr_module=False, model.has_aggr_module=True, model.aggr_module=gcn/gat/gin/graphsage/bnt.
  On the other hand, when training the Dual-pathway model, set model.has_nonaggr_module=True, model.has_aggr_module=True, model.aggr_module=gat.

- **pretrain_lower_epoch**, only used for the proposed Dual-pathway model. Number of independent training epochs for the GAT pathway of the proposed Dual-pathway model.

More running parameters could be referred to the `/source/conf` folders and the provided running scripts for each part of the study.

