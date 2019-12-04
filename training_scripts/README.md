### Training Script Arguments

### Training in De Novo - From Scratch Mode

You can either run  `[notebook run]` cells in `3b) traindenovo.ipynb` to do training within the notebook (which eseentially calls training_scripts/traindenovo.py from notebook). This is the suggested way for training since it is tested and much easier to change parameters but you can also take `training_scripts/traindenovo.py` script and run from the terminal.

#### Parameters

Eventhough notebook's `script` part is very self-explanatory here is a list of parameters you can pass for `traindenovo.py`:

- **data_name:** Data name for experiment, valid args are `notl_brain_mr`, `notl_brain_ct`, `notl_ventricle_mr`, `notl_ventricle_ct`, `atlas_brain_mr`, `atlas_ventricle_mr`

- **sample_size:** Number of random samples for training, default will be None and will use full training data

- **seed:** Random seed for sample_size

- **bs:** Batch size for training. Can be adjusted depending on your GPU memory

- **model_name:** Model architecture config - baseline, valid args are `baseline{1-11}` as defined in experiment_model_dict in `local/models.py`

- **MODEL_NAME:** Model name to save the model, e.g. NOTL_Brain_MR_Baseline_1. This will be the name of to drectory where model files will be saved into e.g. `ATLAS_Brain_MR_Baseline_1/best_of_ATLAS_Brain_MR_Baseline_1.pth`

- **model_dir:** Directory to save the model. If we take the same example above and if `model_dir=atlas_brain_mr_models` then the full path for saved model will be `atlas_brain_mr_models/ATLAS_Brain_MR_Baseline_1/best_of_ATLAS_Brain_MR_Baseline_1.pth`. These structures will be then used in `transfer_learning.yaml` so we suggest keeping structure well for keeping things easy to work with. 

- **loss_func:** Loss function for training, valid args are `bce`, `dice` and `mixed`

- **eps:** Eps value for Adam optimizer

- **epochs:** Number of epochs for training

- **lr:** Learning rate for training


### Training in Weakly Supervised Transfer Learning Mode

You can either run  `[notebook run]` cells in `3c) traintransfer.ipynb` to do training within the notebook (which eseentially calls training_scripts/traintransfer.py from notebook). This is the suggested way for training since it is tested and much easier to change parameters but you can also take `training_scripts/traintransfer.py` script and run from the terminal.

Eventhough notebook's `script` part is very self-explanatory here is a list of parameters you can pass for `traintransfer.py`:

- **data_name:** Data name for experiment, valid args are `notl_brain_mr`, `notl_brain_ct`, `notl_ventricle_mr`, `notl_ventricle_ct`. Eventhough names start with `notl` we fine tune using this data like we use it for training from scratch

- **sample_size:** Number of random samples for training, default will be None and will use full training data

- **seed:** Random seed for sample_size

- **bs:** Batch size for training. Can be adjusted depending on your GPU memory

- **model_name:** Model architecture config - baseline, valid args are `baseline{1-11}` as defined in experiment_model_dict in `local/models.py`

- **MODEL_NAME:** Model name to save the model, e.g. `NOTL_Brain_MR_Baseline_1`. This will be the name of to drectory where model files will be saved into e.g. `ATLAS_Brain_MR_Baseline_1/best_of_ATLAS_Brain_MR_Baseline_1.pth`

- **model_dir:** Directory to save the model. If we take the same example above and if `model_dir=atlas_brain_mr_models` then the full path for saved model will be `atlas_brain_mr_models/ATLAS_Brain_MR_Baseline_1/best_of_ATLAS_Brain_MR_Baseline_1.pth`. These structures will be then used in `transfer_learning.yaml` so we suggest keeping structure well for keeping things easy to work with. 

- **loss_func:** Loss function for training, valid args are `bce`, `dice` and `mixed`

- **TASK:** Task as defined in `transfer_learning.yaml`, valid args are `BRAIN` or `VENTRICLE`, for example if you pick `notl_brain_mr` data for training then task is `BRAIN` as data name suggests

- **MODALITY:** Modality as defined in `transfer_learning.yaml`, valid args are `MR` or `CT`, for example if you pick `notl_brain_mr` data for training then modality is `MR` as data name suggests

- **tl_model_path:** Directory path of the pretrained model inside `experiments` folder. For example, `atlas_brain_mr_models/ATLAS_Brain_MR_Baseline_1` is the directory name where `best_of_ATLAS_Brain_MR_Baseline_1.pth` model is saved. This model name is picked from `transfer_learning.yaml` for the given `MODEL_NAME`, e.g. `NOTL_Brain_MR_Baseline_1`

- **eps:** Eps value for Adam optimizer

- **epochs:** Number of epochs for training

- **lr:** Learning rate for training






