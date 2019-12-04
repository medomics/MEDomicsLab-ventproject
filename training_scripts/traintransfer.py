#export
from fastai2.notebook.core import *
import sys, os

# add local/ package to python path to allow script to access py modules
if not IN_NOTEBOOK: sys.path.insert(0, os.path.abspath("."))

#export
from fastai2.vision.all import *
from fastai2.data.all import *
from local.datasource import *
from local.models import *
from fastai2.torch_core import *
from fastai2.basics import *
from local.trainutils import *
from fastai2.callback.all import *
from fastai2.distributed import *
from time import time
import yaml

#export
@contextmanager
def np_local_seed(seed):
    "numpy local seed - doesn't effect global random state"
    state = np.random.get_state()
    np.random.seed(seed)
    try: yield
    finally: np.random.set_state(state)

#export
class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., fname='model', every_epoch=False, add_save=None, with_opt=False):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        store_attr(self, 'fname,every_epoch,add_save,with_opt')

    def _save(self, name):
        self.learn.save(name, with_opt=self.with_opt)
        if self.add_save is not None:
            with self.add_save.open('wb') as f: self.learn.save(f, with_opt=self.with_opt)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.every_epoch: self._save(f'{self.fname}_{self.epoch}')
        else: #every improvement
            super().after_epoch()
            if self.new_best: self._save(f'{self.fname}')

# loading is a problem in distributed
#     def on_train_end(self, **kwargs):
#         "Load the best model."
#         if not self.every_epoch: self.learn.load(f'{self.fname}')

#export 
@call_parse
def main(
    gpu:Param("GPU to run on", str)=None,
    data_name:Param("Data name for experiment", str)="notl_brain_mr",
    sample_size:Param("Random samples for training, default None - full", int)=None,
    seed:Param("Random seed for sample_size", int)=None,
    bs:Param("Batch size for training", int)=2,
    model_name:Param("Model architecture config - baseline*", str)="baseline1",
    MODEL_NAME:Param("Model name to save the model", str)="TL_Brain_MR_Baseline_1",
    model_dir:Param("Directory to save model", str)="tl_brain_mr_models",
    loss_func:Param("Loss function for training", str)='dice',
    TASK:Param("Task defined for transfer learning in tl.yaml", str)='BRAIN',
    MODALITY:Param("Modality defined for transfer learning in tl.yaml", str)='MR',
    tl_model_path:Param("Relative model path", str)="atlas_brain_mr_models/ATLAS_Brain_MR_Baseline_1", 
    eps:Param("Eps value for Adam optimizer", float)=1e-8,
    epochs:Param("Number of epochs for training", int)=2,
    lr:Param("Learning rate for training", float)=0.1):
    
    "Distributed de novo training - aka. from scratch"
    import os; print(os.getcwd())

    gpu = setup_distrib(gpu)
    n_gpus, gpu_rank = num_distrib(), rank_distrib()

    # data
    dsource = datasource_dict[data_name]()
    if sample_size:
        with np_local_seed(seed):
            dsource.splits[0] = L(np.random.choice(dsource.splits[0], sample_size))  # subsample training
    dbunch = dsource.databunch(after_batch=[Cuda()], bs=bs)
    if len(dbunch.dls) == 4: pass
    elif len(dbunch.dls) == 2: pass
    else: raise Exception(f"DataSource should have either 2 or 4 subsets, but have {len(dsource.splits)}")

    # model
    m = experiment_model_dict[model_name]()
    apply_leaf(m, partial(my_cond_init, func=nn.init.kaiming_normal_))
    
    # callbacks
    save_model_cb = SaveModelCallback(monitor='dice_score', comp=np.greater, every_epoch=False,
                        fname=f'best_of_{MODEL_NAME}')
    callbacks = [TerminateOnNaNCallback(), save_model_cb]        
    
    # learn
    split_func = model_split_dict[model_name]
    lf = loss_dict[loss_func]
    opt_func = partial(Adam, eps=eps)
    learn = Learner(dbunch, m, lf, metrics=[dice_score], opt_func=opt_func,
                    path=Path('experiments'),
                    model_dir=Path(model_dir)/MODEL_NAME, cbs=callbacks, splitter=split_func)
    learn.to_fp16()

    # load pretrained
    with open(os.environ.get('YAML_TL', 'transfer_learning.yaml')) as f: 
        tl = yaml.load(f.read(), yaml.FullLoader)
    path, model_dir = learn.path, learn.model_dir
    tl_model_name = tl[TASK][MODALITY][MODEL_NAME]
    learn.path, learn.model_dir = path, tl_model_path
    learn.load(tl_model_name);
    learn.path, learn.model_dir = path, model_dir
    
    # distributed
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu)    

    # fine tuning - transfer learning
    n_groups = len(learn.opt.param_groups)
    tl_epochs = array([epochs]*n_groups)//np.power(2, array(list(range(n_groups))))
    tl_epochs = np.clip(tl_epochs,1,np.inf).astype(int)
    for i, _epochs in zip(range(1, n_groups+1), tl_epochs):
        if not int(gpu): print(f"Freezing to param group: {-i} and training for {_epochs} epochs")
        learn.freeze_to(-i)
        learn.fit_one_cycle(_epochs, slice(lr), cbs=callbacks)
    


    # evaluate
    if not gpu_rank:
        if len(dbunch.dls) == 4: 
            learn.load(f'best_of_{MODEL_NAME}');
            learn.cbs = [cb for cb in learn.cbs if not isinstance(cb, TrackerCallback) and
                                               not isinstance(cb, TerminateOnNaNCallback)]        
            test1_eval, test2_eval = learn.validate(2), learn.validate(3)
            eval_dir = f"test_results/{model_dir}"
            os.makedirs(eval_dir, exist_ok=True)
            save_fn = f"{eval_dir}/{str(int(time()))}.txt"
            with open(save_fn, 'w') as f: f.write(str([test1_eval, test2_eval]))
