<h1>"Exact, tractable Gauss-Newton optimization in deep reversible architectures reveal poor generalization"</h1>

Repository for the paper "Exact, tractable Gauss-Newton optimization in deep reversible architectures reveal poor generalization", Davide Buffelli\*, Jamie McGowan\*, Wangkun Xu, Alexandru Cioba, Da-shan Shiu, Guillaume Hennequin, Alberto Bernacchia, Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

## Installation
```
# For our experiments we use Python 3.10 and conda, but you should be able to use also other configurations
conda create --name gn python=3.10
conda activate gn

# Install PyTorch, here you can find instructins for your particular configuration: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other packages
pip install einops tqdm hyperopt tensorboard
pip install -U "ray[tune]"

# Setup paths
. ./env.sh
```

## Reproducing Training Curves

The config files are stored for the RevMLP experiments on MNIST and CIFAR that can be found in the paper are stored in `research/RevMLP/config/`.

Inside each configuration file there are some important paths to be defined:
- `Setup > Data > path`: is the location in which you would like to have the dataset stored (the training script will download the datasets from the web on the first execution)
- `Output > Results > output_dir`: is the location in which the training script will save the logs (the logs will be in csv, json, and tensorboard format) and the weights at every epoch (keep in mind that these files can be quite large)

The training can be launched into via:
```
cd research/RevMLP
python3 train.py --config config/config_file_name.yaml
```

if you want to resume a training from a certain checkpoint, you can modify the configuration file and add the path to the checkpoints in the
`Runtime > Train > checkpoint_path` field.

## Reproducing Other Results

### Results on Average Loss Change per Iteration
These results are logged when training the model. They can be found in the csv and json files (under the kys "Percentage loss change per batch") generated during training.

### CKA Results
The script for computing the CKA similarities between start and end of training can be launched as follows:
```
cd research/RevMLP
python3 cka_og.py --config path_to_folder_with_logs_and_weights_for_run_of_interest
```
The script will output a csv file saved in the same folder passed to the script.

The script for computing the CKA similarities between representations learned by different optimizers can be launched as follows:
```
cd research/RevMLP
python3 cka_similarities_cross_opt.py --config path_to_folder_with_logs_for_all_optimizers
```
The script will output a csv file saved in the same folder passed to the script.

### NTK Results
The script for computing the evolution of the NTK similarity during training can be launched as follows:
```
cd research/RevMLP
python3 compute_ntk.py --config path_to_folder_with_logs_and_weights_for_run_of_interest
```
The script will output a csv file saved in the same folder passed to the script.

### Weight Space Results
The script for computing the change in weight (norm and cosine similarity) between start and end of training can be launched as follows:
```
cd research/RevMLP
python3 compute_weight_diff.py --config path_to_folder_with_logs_and_weights_for_run_of_interest
```
The script will output a csv file saved in the same folder passed to the script.

## Citation
```
@inproceedings{
buffelli2024exact,
title={Exact, Tractable Gauss-Newton Optimization in Deep Reversible Architectures Reveal Poor Generalization},
author={Davide Buffelli and Jamie McGowan and Wangkun Xu and Alexandru Cioba and Da-shan Shiu and Guillaume Hennequin and Alberto Bernacchia},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=p37NlKi9vl}
}
```
