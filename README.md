<<<<<<< HEAD
# $Robustness Testing:$ Vulnerability Mining for the RAI2 Auditing Framework

This repository is dedicated to open-sourcing the experimental code for the vulnerability mining (attack scenarios) 
of the auditing framework proposed in our NDSS'23 paper, *"RAI2: Responsible Identity Audit Governing the Artificial 
Intelligence"* in terms of model ownership protection. If you find our open-source code helpful, please give us a star, thank you! The specific workflow will be provided below.



## Environment setting

The code is tested on Python 3.10.12, Pytorch 2.0.1, CUDA 11.8.
The packages used in our project is provided in ```requirements.txt``` for reference.


## File and Folder Description

- `conf/global_settings.py`: parameters used for all scripts include paths of data and model, etc.
- `models/`: model architecture files,this part of the work has not been developed yet.
- `dataset_preparation/`: jupyter notebooks for preprocess datasets,We constructed five types of dataset reconstruction transformations for subsequent evaluation.
- `dataset_similarity/`: notebooks for reproduce the results of dataset similarity estimation.
- `model_similarity`: notebooks for reproduce the results of model similarity estimation,this part of the work has not been developed yet.
- `train_cv.py`: script to train models on CIFAR-10/100 and Tiny-ImageNet,each file corresponds to a type of reconstruction transformation to maintain parallelism in the experiments.
- `utils.py`: script containing auxillary functions like data loading, network initialization and training, etc.each file corresponds to a type of reconstruction transformation to maintain parallelism in the experiments.

### Workflow Introduction:

1. **Set Global Paths**  
   First, please modify variables in the format of `{}_PATH` in `conf/global_settings.py` to specify the storage locations for your models and data. Here, we also set up multiple global paths in a parallel structure, which improves efficiency and provides a clearer structure during the preparation of data and training of models.

2. **Dataset Preparation**  
   Use the `CIFARSplitData.ipynb` and `TinyImagenetSplitData.ipynb` notebooks in the `dataset_preparation` directory to split the CIFAR-10/CIFAR-100 and TinyImagenet-200 datasets. This step mainly involves pre-constructing the subsets `subset_1` and `subset_2` for the roles of the victim and the attacker.

3. **Constructing Attack Datasets**  
   Use the `DataSimilarityNBS.ipynb` notebook in the `dataset_preparation` directory to create datasets used by attackers after stealing data under corresponding attack methods. In this step, the code allows you to simulate the intensity of attacks by adjusting the proportions of datasets and the transformation strength.

4. **Training Surrogate Models**  
   Train surrogate models for attackers under different datasets and attack methods using the following scripts:
   - `train_cv.py` 
   - `train_cvnoise.py`
   - `train_cvshear.py`
   - `train_cvbrightness.py`
   - `train_cvtranslate.py`

Each file corresponds to a type of dataset reconstruction attack.

   To run the scripts, refer to the parameter descriptions provided in them. Here is an example:
   ```bash
   python train_cv.py -net resnet18 -dataset cifar10 -subset 1 -inter_propor 0.0 -copy_id 0 -gpu_id 0   victim
   python train_cv.py -net vgg13 -dataset cifar10 -subset 2 -inter_propor 0.0 -copy_id 0 -gpu_id 0     attacker
   ```

   The training settings and hyperparameters for surrogate models are exactly the same as those used in the RAI2 work. The purpose of this is to control variables and ensure a consistent basis for subsequent robustness evaluations.

5. **Evaluating Model Accuracy**  
   After training the surrogate models, use the `evaluate_accuracy.py` script to infer Top-1 and Top-5 model accuracies. This ensures that the attacker does not lose their motivation to launch attacks or pay an unacceptable cost for the attacks.

6. **Experimental Data Analysis**  
   Use the scripts in the `dataset_similarity` directory to analyze the experimental data. Among them:
   - The script `similarity_cv_predict.py` generates model outputs as intermediate results to facilitate faster result analysis in notebooks.
   - The main analysis is done in the `InferSimilarity.ipynb` notebook.

   For analyzing the data, the evaluation metrics, methods, and parameter settings strictly follow those used in the RAI2 experimental work. The goal is to 100% replicate their methodology and control variables for consistency.

---

### Additional Information:

If you are not familiar with the auditing mechanism proposed in the RAI2 work, please refer to the paper:  
**"Dong T, Li S, Chen G, et al. RAI2: Responsible Identity Audit Governing the Artificial Intelligence [C] // NDSS. 2023."**  
or the source code of their work: [https://github.com/chichidd/RAI2.git](https://github.com/chichidd/RAI2.git).
# Robustness-Testing-of-RAI2
>>>>>>> 
