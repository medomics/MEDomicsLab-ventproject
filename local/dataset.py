#AUTOGENERATED! DO NOT EDIT! File to edit: dev/2) dataset.ipynb (unless otherwise specified).

__all__ = ['data_paths', 'csv_paths', 'ct_metadf', 'ct_test2_metadf', 'mr_metadf', 'mr_test2_metadf',
           'val_test_patients', 'mr_val', 'mr_test1', 'ct_val', 'ct_test1', 'mr_val_suids', 'mr_test1_suids',
           'mr_test2_suids', 'mr_train_suids', 'ct_val_suids', 'ct_test1_suids', 'ct_test2_suids', 'ct_train_suids',
           'filter_image', 'filter_brain_mask', 'filter_ventricles_mask', 'filter_mr_train', 'filter_mr_val',
           'filter_mr_test1', 'filter_mr_test2', 'filter_ct_train', 'filter_ct_val', 'filter_ct_test1',
           'filter_ct_test2', 'sort_by_suid', 'BrainDataset3D', 'files', 'mr_image_fnames', 'mr_train_image_fnames',
           'mr_val_image_fnames', 'mr_test1_image_fnames', 'mr_test2_image_fnames', 'mr_brain_mask_fnames',
           'mr_train_brain_mask_fnames', 'mr_val_brain_mask_fnames', 'mr_test1_brain_mask_fnames',
           'mr_test2_brain_mask_fnames', 'mr_ventricles_mask_fnames', 'mr_train_ventricles_mask_fname',
           'mr_val_ventricles_mask_fname', 'mr_test1_ventricles_mask_fname', 'mr_test2_ventricles_mask_fname',
           'get_notl_brain_mr_data', 'get_notl_ventricle_mr_data', 'files', 'ct_image_fnames', 'ct_train_image_fnames',
           'ct_val_image_fnames', 'ct_test1_image_fnames', 'ct_test2_image_fnames', 'ct_brain_mask_fnames',
           'ct_train_brain_mask_fnames', 'ct_val_brain_mask_fnames', 'ct_test1_brain_mask_fnames',
           'ct_test2_brain_mask_fnames', 'ct_ventricles_mask_fnames', 'ct_train_ventricles_mask_fname',
           'ct_val_ventricles_mask_fname', 'ct_test1_ventricles_mask_fname', 'ct_test2_ventricles_mask_fname',
           'get_notl_brain_ct_data', 'get_notl_ventricle_ct_data', 'files', 'atlas_image_fnames',
           'atlas_brain_mask_fnames', 'atlas_ventricles_mask_fnames', 'get_atlas_brain_mr_data',
           'get_atlas_ventricle_mr_data', 'data_dict']

#Cell
from fastai2.basics import *
from fastai2.vision import *

#Cell
data_paths = types.SimpleNamespace(
    ATLAS_OUTPUT="/home/turgutluk/data/ventricles_data/atlas",
    MR_OUTPUT="/home/turgutluk/data/ventricles_data/mr",
    CT_OUTPUT="/home/turgutluk/data/ventricles_data/ct")

#Cell
csv_paths = types.SimpleNamespace(
    CT_META = '/home/turgutluk/data/ventricles_data/csvs/CT_PATH_META.csv',
    CT_TEST2_META = '/home/turgutluk/data/ventricles_data/csvs/CT_TEST2_PATH_META.csv',
    MR_META = '/home/turgutluk/data/ventricles_data/csvs/MR_PATH_META.csv',
    MR_TEST2_META = '/home/turgutluk/data/ventricles_data/csvs/MR_TEST2_PATH_META.csv'
)

#Cell
ct_metadf = pd.read_csv(csv_paths.CT_META,low_memory=False)
ct_test2_metadf = pd.read_csv(csv_paths.CT_TEST2_META,low_memory=False)
mr_metadf = pd.read_csv(csv_paths.MR_META,low_memory=False)
mr_test2_metadf = pd.read_csv(csv_paths.MR_TEST2_META,low_memory=False)

#Cell
val_test_patients = types.SimpleNamespace(
mr_val = {'ANON61382','ANON55375','ANON85534','ANON54218','ANON24182','ANON14135','ANON49037',
          'ANON66932','ANON10465','ANON39801','ANON14447','ANON42229','ANON99458','ANON36946',
          'ANON16732'},
mr_test1 = {'ANON78381','ANON38662','ANON78219','ANON65248','ANON98217','ANON22366', 'ANON53486',
            'ANON80073','ANON93045','ANON26348','ANON72855','ANON60446','ANON28622','ANON60751',
            'ANON41567'},
ct_val = {'ANON85656','ANON24135','ANON45434','ANON53464','ANON50198','ANON86095','ANON47701',
          'ANON21818','ANON13928','ANON45164','ANON57908','ANON10634','ANON37574','ANON13983',
          'ANON39193','ANON52842','ANON83901','ANON34509','ANON14150','ANON70712','ANON36668',
          'ANON86933','ANON69869','ANON55750'},
ct_test1 = {'ANON95021', 'ANON17272', 'ANON45950', 'ANON71219', 'ANON84614', 'ANON22673',
            'ANON65837', 'ANON51808', 'ANON24224'})

#Cell
# MR train, val, test1 and test2 StudyInstance UIDs
mr_val_suids = np.unique(mr_metadf[mr_metadf.PatientID.isin(val_test_patients.mr_val)].StudyInstanceUID)
mr_test1_suids = np.unique(mr_metadf[mr_metadf.PatientID.isin(val_test_patients.mr_test1)].StudyInstanceUID)
mr_test2_suids = np.unique(mr_test2_metadf.StudyInstanceUID)
mr_train_suids = (np.unique(mr_metadf[~mr_metadf.StudyInstanceUID
                                      .isin(np.concatenate([mr_val_suids, mr_test1_suids]))]
                                      .StudyInstanceUID))

#Cell
# CT train, val, test1 and test2 StudyInstance UIDs
ct_val_suids = np.unique(ct_metadf[ct_metadf.PatientID.isin(val_test_patients.ct_val)].StudyInstanceUID)
ct_test1_suids = np.unique(ct_metadf[ct_metadf.PatientID.isin(val_test_patients.ct_test1)].StudyInstanceUID)
ct_test2_suids = np.unique(ct_test2_metadf.StudyInstanceUID)
ct_train_suids = (np.unique(ct_metadf[~ct_metadf.StudyInstanceUID
                                      .isin(np.concatenate([ct_val_suids, ct_test1_suids]))]
                                      .StudyInstanceUID))

#Cell
def filter_image(o): return 'image' in o.name
def filter_brain_mask(o): return 'brain' in o.name
def filter_ventricles_mask(o): return 'ventricles' in o.name

#Cell
def filter_mr_train(o): return o.name.split("_")[0] in mr_train_suids
def filter_mr_val(o): return o.name.split("_")[0] in mr_val_suids
def filter_mr_test1(o): return o.name.split("_")[0] in mr_test1_suids
def filter_mr_test2(o): return o.name.split("_")[0] in mr_test2_suids

def filter_ct_train(o): return o.name.split("_")[0] in ct_train_suids
def filter_ct_val(o): return o.name.split("_")[0] in ct_val_suids
def filter_ct_test1(o): return o.name.split("_")[0] in ct_test1_suids
def filter_ct_test2(o): return o.name.split("_")[0] in ct_test2_suids

#Cell
def sort_by_suid(l): return sorted(l, key=lambda o: o.name.split("_")[0])

#Cell
class BrainDataset3D:
    def __init__(self, image_fnames, mask_fnames):
        self.image_fnames = sort_by_suid(image_fnames)
        self.mask_fnames = sort_by_suid(mask_fnames)

    def __getitem__(self, idx):
        image_voxel = torch.load(self.image_fnames[idx])
        mask_voxel = torch.load(self.mask_fnames[idx])
        return image_voxel[None,...], mask_voxel

    def __len__(self): return len(self.image_fnames)

#Cell
files = get_files(data_paths.MR_OUTPUT, extensions=['.pt'])

#Cell
# mr train images
mr_image_fnames = list(filter(filter_image, files))
mr_train_image_fnames = list(filter(filter_mr_train, mr_image_fnames))
mr_val_image_fnames = list(filter(filter_mr_val, mr_image_fnames))
mr_test1_image_fnames = list(filter(filter_mr_test1, mr_image_fnames))
mr_test2_image_fnames = list(filter(filter_mr_test2, mr_image_fnames))

#Cell
# mr brain masks
mr_brain_mask_fnames = list(filter(filter_brain_mask, files))
mr_train_brain_mask_fnames = list(filter(filter_mr_train, mr_brain_mask_fnames))
mr_val_brain_mask_fnames = list(filter(filter_mr_val, mr_brain_mask_fnames))
mr_test1_brain_mask_fnames = list(filter(filter_mr_test1, mr_brain_mask_fnames))
mr_test2_brain_mask_fnames = list(filter(filter_mr_test2, mr_brain_mask_fnames))

#Cell
# mr ventricle masks
mr_ventricles_mask_fnames = list(filter(filter_ventricles_mask, files))
mr_train_ventricles_mask_fname = list(filter(filter_mr_train, mr_ventricles_mask_fnames))
mr_val_ventricles_mask_fname = list(filter(filter_mr_val, mr_ventricles_mask_fnames))
mr_test1_ventricles_mask_fname = list(filter(filter_mr_test1, mr_ventricles_mask_fnames))
mr_test2_ventricles_mask_fname = list(filter(filter_mr_test2, mr_ventricles_mask_fnames))

#Cell
def get_notl_brain_mr_data():
    train_ds = BrainDataset3D(mr_train_image_fnames, mr_train_brain_mask_fnames)
    valid_ds = BrainDataset3D(mr_val_image_fnames, mr_val_brain_mask_fnames)
    test1_ds = BrainDataset3D(mr_test1_image_fnames, mr_test1_brain_mask_fnames)
    test2_ds = BrainDataset3D(mr_test2_image_fnames, mr_test2_brain_mask_fnames)
    return train_ds,valid_ds,test1_ds,test2_ds

def get_notl_ventricle_mr_data():
    train_ds = BrainDataset3D(mr_train_image_fnames, mr_train_ventricles_mask_fname)
    valid_ds = BrainDataset3D(mr_val_image_fnames, mr_val_ventricles_mask_fname)
    test1_ds = BrainDataset3D(mr_test1_image_fnames, mr_test1_ventricles_mask_fname)
    test2_ds = BrainDataset3D(mr_test2_image_fnames, mr_test2_ventricles_mask_fname)
    return train_ds,valid_ds,test1_ds,test2_ds

#Cell
files = get_files(data_paths.CT_OUTPUT, extensions=['.pt'])

#Cell
# ct train images
ct_image_fnames = list(filter(filter_image, files))
ct_train_image_fnames = list(filter(filter_ct_train, ct_image_fnames))
ct_val_image_fnames = list(filter(filter_ct_val, ct_image_fnames))
ct_test1_image_fnames = list(filter(filter_ct_test1, ct_image_fnames))
ct_test2_image_fnames = list(filter(filter_ct_test2, ct_image_fnames))

#Cell
# ct brain masks
ct_brain_mask_fnames = list(filter(filter_brain_mask, files))
ct_train_brain_mask_fnames = list(filter(filter_ct_train, ct_brain_mask_fnames))
ct_val_brain_mask_fnames = list(filter(filter_ct_val, ct_brain_mask_fnames))
ct_test1_brain_mask_fnames = list(filter(filter_ct_test1, ct_brain_mask_fnames))
ct_test2_brain_mask_fnames = list(filter(filter_ct_test2, ct_brain_mask_fnames))

#Cell
# mr ventricle masks
ct_ventricles_mask_fnames = list(filter(filter_ventricles_mask, files))
ct_train_ventricles_mask_fname = list(filter(filter_ct_train, ct_ventricles_mask_fnames))
ct_val_ventricles_mask_fname = list(filter(filter_ct_val, ct_ventricles_mask_fnames))
ct_test1_ventricles_mask_fname = list(filter(filter_ct_test1, ct_ventricles_mask_fnames))
ct_test2_ventricles_mask_fname = list(filter(filter_ct_test2, ct_ventricles_mask_fnames))

#Cell
def get_notl_brain_ct_data():
    train_ds = BrainDataset3D(ct_train_image_fnames, ct_train_brain_mask_fnames)
    valid_ds = BrainDataset3D(ct_val_image_fnames, ct_val_brain_mask_fnames)
    test1_ds = BrainDataset3D(ct_test1_image_fnames, ct_test1_brain_mask_fnames)
    test2_ds = BrainDataset3D(ct_test2_image_fnames, ct_test2_brain_mask_fnames)
    return train_ds,valid_ds,test1_ds,test2_ds

def get_notl_ventricle_ct_data():
    train_ds = BrainDataset3D(ct_train_image_fnames, ct_train_ventricles_mask_fname)
    valid_ds = BrainDataset3D(ct_val_image_fnames, ct_val_ventricles_mask_fname)
    test1_ds = BrainDataset3D(ct_test1_image_fnames, ct_test1_ventricles_mask_fname)
    test2_ds = BrainDataset3D(ct_test2_image_fnames, ct_test2_ventricles_mask_fname)
    return train_ds,valid_ds,test1_ds,test2_ds

#Cell
files = get_files(data_paths.ATLAS_OUTPUT, extensions=['.pt'])

#Cell
atlas_image_fnames = list(filter(filter_image, files))
atlas_brain_mask_fnames = list(filter(filter_brain_mask, files))
atlas_ventricles_mask_fnames = list(filter(filter_ventricles_mask, files))

#Cell
def get_atlas_brain_mr_data():
    train_ds = BrainDataset3D(atlas_image_fnames, atlas_brain_mask_fnames)
    valid_ds = BrainDataset3D(mr_val_image_fnames, mr_val_brain_mask_fnames)
    test1_ds = None
    test2_ds = None
    return train_ds,valid_ds,test1_ds,test2_ds

def get_atlas_ventricle_mr_data():
    train_ds = BrainDataset3D(atlas_image_fnames, atlas_ventricles_mask_fnames)
    valid_ds = BrainDataset3D(mr_val_image_fnames, mr_val_ventricles_mask_fname)
    test1_ds = None
    test2_ds = None
    return train_ds,valid_ds,test1_ds,test2_ds

#Cell
data_dict = {
    'notl_brain_mr': get_notl_brain_mr_data,
    'notl_brain_ct': get_notl_brain_ct_data,
    'atlas_brain_mr': get_atlas_brain_mr_data,
    'notl_ventricle_mr': get_notl_ventricle_mr_data,
    'notl_ventricle_ct': get_notl_ventricle_ct_data,
    'atlas_ventricle_mr': get_atlas_ventricle_mr_data,
}
