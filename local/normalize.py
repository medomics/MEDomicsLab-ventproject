#AUTOGENERATED! DO NOT EDIT! File to edit: dev/1c) normalization.ipynb (unless otherwise specified).

__all__ = ['input_paths', 'output_paths', 'main']

#Cell
from fastai2.medical.imaging_roi import *
from fastai2.medical.imaging import dicom_windows
from fastai2 import *
from fastai2.torch_core import *
from fastai2.core import *
from fastai2.basics import *

#Cell
def _normalize(t, mean, std):
    "normalization func"
    t = torch.clamp((t - mean) / std, -5, 5)
    _min, _max = torch.min(t), torch.max(t)
    return (t - _min) / (_max - _min)

#Cell
def _normalize_images_and_save(o):
    "Normalizes individual images to 0-1 scale and save"
    # read image
    t = torch.load(o)
    # normalize
    std,mean = torch.std_mean(t)
    t = _normalize(t, mean, std)
    # save
    p = o.parent
    suid = o.name.split('_')[0]
    torch.save(t, p/f"{suid}_image_normalized.pt")

#Cell
def _normalize_skull_stripped_images_and_save(o):
    "Normalizes individual skull stripped images to 0-1 scale and save"
    # read image and mask
    t = torch.load(o)
    p = o.parent
    suid = o.name.split('_')[0]
    msk = torch.load(p/f"{suid}_brain_mask.pt")
    # normalize
    std,mean = torch.std_mean(t[msk.bool()])
    t = _normalize(t, mean, std)*msk
    # save
    torch.save(t, p/f"{suid}_skull_stripped_image_normalized.pt")

#Cell
import yaml
with open(os.environ.get('YAML_DATA', '../data.yaml')) as f: data_config = yaml.load(f.read(), yaml.FullLoader)

# define input and output paths
input_paths = types.SimpleNamespace(
    ATLAS_PATH=data_config['input']['ATLAS_PATH'],
    MR_PATH=data_config['input']['MR_PATH'],
    CT_PATH=data_config['input']['CT_PATH'],
    MR_TEST2_PATH=data_config['input']['MR_TEST2_PATH'],
    CT_TEST2_PATH=data_config['input']['CT_TEST2_PATH'],
)

output_paths = types.SimpleNamespace(
    ATLAS=data_config['output']['ATLAS'],
    MR=data_config['output']['MR'],
    CT=data_config['output']['CT'])

#Cell
from time import perf_counter
@call_parse
def main(output_path:Param("Directory that have data prep results", str)):
    "Read tensors, normalize images and skull stripped images"
    start = perf_counter()

    output_path = Path(output_paths.__dict__[output_path])
    files = get_files(output_path, extensions=['.pt'])
    image_files = [o for o in files if "_".join(o.name.split("_")[1:]) == "image.pt"]
    skull_stripped_image_files = [o for o in files if "_".join(o.name.split("_")[1:]) == "skull_stripped_image.pt"]
    parallel(_normalize_images_and_save, image_files, n_workers=defaults.cpus//2)
    parallel(_normalize_skull_stripped_images_and_save, skull_stripped_image_files, n_workers=defaults.cpus//2)

    end = perf_counter()
    print(f"Total time taken {end-start} seconds")