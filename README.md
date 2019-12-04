## Weakly Supervised Transfer Learning for Ventricle Segmentation v2


In this repo you will find notebooks and scripts which will allow you to find all the code necessary to replicate the experiments conducted in this project. This v2 version aims to be as simple as possible in terms of code readability and project structure. Improvements, suggestions and feedback are welcome.

Main library of the project is a custom forked version of [fastai v2](https://github.com/KeremTurgutlu/fastai2/tree/extend_medical). Initially original code for experiments were written in an older version of PyTorch and Fast.ai. With this migration the code is now more organized and faster for mass consumption. Workflow adapted in this repo is pretty generic and potentially can be used for other similar projects with different data or task with minimal changes. Stay tuned for our next project :)



### Setup

- Follow the **Installation** steps in https://dev.fast.ai/ (including `fastai2.medical.imaging`). This installation step suggest creating a new environment, please do so. Let's say we created a new environment called `ventproject`.
- Activate environment `conda activate ventproject`. Now we are in that environment.
- Clone this project repo 
- Clone the custom forked version of [fastai v2](https://github.com/KeremTurgutlu/fastai2/tree/extend_medical).
- cd into custom fork and do `pip install -e .` this will install that repo as a pip package: `fastai2`


### Configs

In this project you will need to have two `yaml` config files in `configs/` one for preparing data and one for defining transfer learned to pretrained model mapping. 

`data.yaml`: It is needed as we share raw DICOM images and this configuration will allow you to define where to read and where to write. It also will have csv metadata for each modality which has `train`, `valid`, `test1` and `test2` split information for each `StudyInstanceUID`.

#### How to fill `data.yaml`

In general if you check `data.yaml` shared in this repo you will have a good idea on how to fill it jut by looking at it.

- **input:** Defines the input locations for raw DICOM data. Please enter absolute path for:

```
    - ATLAS_PATH: Absolute path for `PICARE_BMETS_Raw_DICOM_Files`
    - MR_PATH: Absolute path for `PICARE_SEGMENTATION_BRAINVENT_MR_V1`
    - CT_PATH: Absolute path for `Training_CT_Raw_DICOM_Files`
    - MR_TEST2_PATH: Absolute path for `Testing_MR_Raw_DICOM_Files`
    - CT_TEST2_PATH: Absolute path for `Testing_CT_Raw_DICOM_Files`
```

- **output:** Defines where the processed DICOM data will be saved, e.g. pytorch tensors for training. It's good to have the following under the same parent directory, e.g. something like `{somepath}/ventproject_data`

```
    - ATLAS: Absolute path for processed atlas data
    - MR: Absolute path for processed mr data
    - CT: Absolute path for processed mr data
```

- **csv_splits:** Defines where the csv split data is located. These csv files are again shared by us. They have train, valid, test1 and test2 information.

```
    - ATLAS:  Absolute path for `atlas_splits_df.csv`
    - MR:  Absolute path for `mr_splits_df.csv`
    - CT:  Absolute path for `ct_splits_df.csv`
```

`transfer_learning.yaml`: Here we define transfer learning model to pretrained model mappings for a given `TASK - MODALITY` combination. This is customizable depending on which of the pretrained models you trained locally so far. By default all of our original mappings are left in this file to give an example.

#### How to fill `transfer_learning.yaml`

Once you pretrain atlas models you can go ahead to define `transfer_learning.yaml` to define transfer learned models to pretrained models mapping for a given task (BRAIN/VENTRICLE) and modality (MR/CT) pair. You can notice from the shared `tansfer_learning.yaml` file in this repo that we map transfer learning experiment name `TL_Brain_MR_Baseline_1` to a pretrained model name `best_of_ATLAS_Brain_MR_Baseline_1` which should be located somewhere as a pretrained pytorch model file as `best_of_ATLAS_Brain_MR_Baseline_1.pth`

If you follow either the notebooks or the scripts you can get a good understanding on how to fill this `yaml` file.

### Setting Environment Variables

Whenever you run any script or notebook two environment variables should be set:

- `YAML_DATA`: Absolute path to `configs/data.yaml`
- `YAML_TL`: Absolute path to `configs/transfer_learning.yaml`

This can be done during conda environment initialisation as [here](https://stackoverflow.com/questions/31598963/how-to-set-specific-environment-variables-when-activating-conda-environment)

In notebooks you will see section **set `data.yaml`** where this is applicable. For scripts these environemnt variables should be set properly before calling them in the same shell environment.



### Notebooks

In this project a technique called `literate programming` is used, meaning that most of the source code (80%) is generated using jupyter notebooks. This allows interactivity, fast development and transparency for the users. 

In this project you have the option to either use the notebooks or the scripts!

- `0) scriptrunner.ipynb`: Implements utilities for running scripts in notebook env. Better left as is.

- `1a) dicom to tensor.ipynb`: Read DICOM data, resample, crop-pad and save. (has a runnable script)

- `1b) skull strip.ipynb`: Skull strip data. (has a runnable script)

- `1c) normalization.ipynb`: Normalize data to (0-1) scale for training. (has a runnable script)

- `2) datasource.ipynb`: Defines fastai v2 `DataSource` (python module)

- `3a) trainutils.ipynb`: Implements training utilities (python module)

- `3b) traindenovo.ipynb`: End-to-end training for de novo mode  (has a runnable script)

- `3c) traintransfer.ipynb`: End-to-end training for weakly supervised transfer learning (has a runnable script)



### Scripts

If you don't want to prepare data using notebooks you use scripts instead that will prepare all data in one go for you. Notebooks are mainly good for understanding what's going on interavtively and potentially changing stuff for exploration purposes. Notebook generated python scripts are located in `local/` with other python modules, training scripts are located in `training_scripts/` and bash scripts are located in `bash_scripts/`.

### Data Prep

### 1) Read -> Convert2Tensor -> Resample -> Save -> Read -> Crop-Pad -> Save

Instead of running `[notebook run]` cells in `1a) dicom to tensor.ipynb` notebook you can use `dicom2tensor.sh` bash script to prepare data. As you may understand from the title this script will `Read -> Conert2Tensor -> Resample -> Save -> Read -> Crop-Pad -> Save` data according to input and output paths defined in `data.yaml`.

Simply run `sh ./dicom2tensor.sh` from same directory where `dicom2tesnsor.sh` script is located.

Logs for this run will be put in `.log/dicom2tensor.log` file

### 2) Read -> Skull Strip -> Save

Instead of running `[notebook run]` cells in `1b) skull strip.ipynb` notebook you can use this `skullstrip.sh` bash script to prepare data. As you may understand from the title this script will `Read -> Skull Strip -> Save` data according to output paths defined in `data.yaml`.

Simply run `sh ./skullstrip.sh` from same directory where `skullstrip.sh` script is located.

Logs for this run will be put in `.log/skullstrip.log` file


### 3) Read -> Normalize -> Save

Instead of running `[notebook run]` cells in `1c) normalization` notebook you can use this `normalize.sh` bash script to prepare data. As you may understand from the title this script will `Read -> Normalize -> Save` data according to output paths defined in `data.yaml`.

Simply run `sh ./normalize.sh` from same directory where `normalize.sh` script is located.

Logs for this run will be put in `.log/normalize.log` file


### 4) All data preparation in one go

Optionally you can run `sh ./preparedata.sh` from same directory where `preparedata.sh` script is located to run steps 1-2-3 in one go.




### Training

For detailed explanation for arguments that can be passed to training scripts please go to `training_scripts/` and read `README.md`. When either `traindenovo,py` or `traintransfer.py` executed an `experiments` folder will be created and all generated model files will saved in it according to defined model paths and names.

### 1) Training in De Novo (from Scratch) Mode

This part is where we can train `ATLAS` pretrained models from scratch as well as `NOTL CT` or `NOTL MR` models for both `BRAIN` and `VENTRICLE` tasks. 

### 2) Training in Weakly Supervised Transfer Learning Mode

This part is where we can finetune `CT` and `MR` models for both `BRAIN` and `VENTRICLE` tasks using pretrained `ATLAS` models.


### How to modify code to try new things!

Will be explained... 












