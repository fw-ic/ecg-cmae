## Modally-Reduced Representation Learning for Multi-Lead ECG Signals

This repository contains the implementation used to train models and produce results for a modally-reduced ECG representation learning investigation reported in our [TS4H Workshop / ICLR 2024 paper](https://iclr.cc/virtual/2024/23547). For an ArXiv version of the paper, see:</br> 
* [Modally Reduced Representation Learning of Multi-Lead ECG Signals through Simultaneous Alignment and Reconstruction](https://arxiv.org/abs/2405.19359)

## ECG Data Used
To get the ECG data used in this paper, you may try:

* wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/training/

## Main Files:
Here's a brief description of the main files:

* ecg_id.ipynb : Notebook for Authentication downstream task
* observe_results_correlated.ipynb : Notebook for signal reconstruction and visualization
* ecg_quality_assement.ipynb : Notebook for ECG quality assement
* pretrainCorrelated.py : pretraining code
* MAE1DCorrelated.py : model architecture         
* refiner_unet.py : refine the reconstructed signal
* MAEBank.py : distributed model 
* myocardial.ipynb : Notebook for MI downstream task           
* util/data_handling.py : data handling utl
* util/ecg_dataset.py : dataset util
* util/pos_embed.py : position embedding util
* util/data_processing.py : simple data processing util

## WARNING
* pytorch<=1.12 is a must
