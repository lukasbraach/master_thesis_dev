<div align="center">

# master_thesis_dev

Code of my master thesis implementation

The setup of this repository was adapted from https://github.com/ashleve/lightning-hydra-template

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/lukasbraach/master_thesis_dev"><img alt="Code" src="https://img.shields.io/badge/-GitHub-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This repository contains the implementation of my master thesis titled "Knowledge Transfer: Progressing from Gesture Recognition to the Transcription of Sign Languages."

The project focuses on applying transformer architectures, specifically Video-Vision-Transformers (ViViTs), 
to sign language recognition using datasets like RWTH Phoenix Weather 2014. 
It includes training and evaluation scripts, as well as experiment configurations for various model setups and ablation studies.

Have a look at my HuggingFace account for the datasets and model checkpoints used in this project:
https://huggingface.co/lukasbraach

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Before running, make sure to check-out the source datasets and modify the dataset_source paths 
in the configs/data/ directory.
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# download the RWTH Phoenix Weather 2014 dataset
git clone https://huggingface.co/datasets/lukasbraach/rwth_phoenix_weather_2014 /data/1braach/rwth_phoenix_weather_2014

# download the Bundestag Barrierefrei dataset
git clone https://huggingface.co/datasets/lukasbraach/bundestag_slr /data/1braach/bundestag_slr

# if the data path is different, adapt the dataset_source paths in the configs/data/ directory
nano configs/data/bundestag_slr_pretrain.yaml && \
nano configs/data/rwth_phoenix_2014_pre.yaml && \
nano configs/data/rwth_phoenix_2014.yaml
```

Train model with default configuration
```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

You can upload your model to the HuggingFace model hub by running the following command:

```bash
python src/save_pretrained.py experiment=rwth_videomae_finetune ckpt_path="your_checkpoint_path" +push_hf_hub="your_account/videomae_rwth_pretrain"
```

## Experiment Configurations

The repository includes several experiment configurations located in the configs/experiment/ directory. 
Each configuration file (.yaml) specifies different setups and hyperparameters for the experiments, enabling the replication of results
and facilitating further research.

Experiment Configurations:

1.	[rwth_from_scratch.yaml](configs%2Fexperiment%2Frwth_from_scratch.yaml):
This configuration file contains the default parameters for training the baseline model. 
It is a good starting point for understanding the basic setup and workflow.

2. [rwth_spatiotemporal_finetune.yaml](configs%2Fexperiment%2Frwth_spatiotemporal_finetune.yaml):
Configuration for training the custom encoder architecture. It integrates multiple components like DINOv2 and Wav2Vec 2.0 to 
assess their combined performance in sign language recognition tasks.

3.	[rwth_videomae_finetune.yaml](configs%2Fexperiment%2Frwth_videomae_finetune.yaml):
This file is used for ablation studies with the VideoMAE model. It helps to evaluate the effectiveness of using 
VideoMAE as a spatiotemporal encoder in comparison to other setups.

4.  [spatiotemporal_pre_training.yaml](configs%2Fexperiment%2Fspatiotemporal_pre_training.yaml):
This configuration utilizes the DINOv2 + Wav2Vec 2.0 model for pre-training, focusing on spatial vision feature extraction. 
It is designed to explore the impact of self-supervised pre-training on model performance, using the Bundestag Barrierefrei dataset.

5. [spatiotemporal_pre_training.yaml](configs%2Fexperiment%2Fspatiotemporal_pre_training.yaml):
After experiment 4 has run, another round of fine-tuning is done on the RWTH Phoenix Weather 2014 dataset.

6. [videomae_pre_training.yaml](configs%2Fexperiment%2Fvideomae_pre_training.yaml):
This configuration utilizes the VideoMAE model for pre-training, focusing on spatialtemporal feature extraction. 
It is designed to explore the impact of self-supervised pre-training on model performance, using the Bundestag Barrierefrei dataset.

7. [videomae_rwth_pre_training.yaml](configs%2Fexperiment%2Fvideomae_rwth_pre_training.yaml):
After experiment 6 has run, another round of fine-tuning is done on the RWTH Phoenix Weather 2014 dataset.

8. [langdecoder_pre_training.yaml](configs%2Fexperiment%2Flangdecoder_pre_training.yaml):
Pre-training of the decoder on transcriptions from the RWTH Phoenix Weather 2014 dataset.