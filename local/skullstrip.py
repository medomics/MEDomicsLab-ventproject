#AUTOGENERATED! DO NOT EDIT! File to edit: dev/1b) skull strip.ipynb (unless otherwise specified).

__all__ = ['input_paths', 'output_paths', 'main']

#Cell
from fastai2.medical.imaging_roi import *
from fastai2.medical.imaging import dicom_windows
from fastai2 import *
from fastai2.torch_core import *
from fastai2.core import *
from fastai2.basics import *

#Cell
def _skull_strip_and_save(o):
    "skull strips image using brain mask and saves new image"
    p = o.parent
    suid = o.name.split('_')[0]
    image = torch.load(p/f"{suid}_image.pt")
    brain_mask = torch.load(p/f"{suid}_brain_mask.pt")
    skull_stripped_image = image*brain_mask
    torch.save(skull_stripped_image, p/f"{suid}_skull_stripped_image.pt")

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
    "Read tensors, skull strip images and save it as a new image"
    start = perf_counter()

    output_path = Path(output_paths.__dict__[output_path])
    files = get_files(output_path, extensions=['.pt'])
    parallel(_skull_strip_and_save, files, n_workers=defaults.cpus//2)

    end = perf_counter()
    print(f"Total time taken {end-start} seconds")