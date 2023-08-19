.### Data Download

wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/training/


## WARNING

pytorch<=1.12 is a must


## Codes:

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
