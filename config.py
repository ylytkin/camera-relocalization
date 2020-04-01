from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import image as img
from tqdm import tqdm

__all__ = [
    'ROOT_DIR',
    'EXTERNAL_DATA_DIR',
    'load_train_data',
]

ROOT_DIR = Path(__file__).absolute().parent

EXTERNAL_DATA_DIR = Path('/home/yuralytkin/Development/data/camera_relocalization_sample_dataset')


def load_train_data(external_data_dir: Path = EXTERNAL_DATA_DIR, rescaling_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Load data for model training.
    
    The images are loaded with halved resolution and without the alpha channel.
    """
    
    images_dir = external_data_dir / 'images'
    assert images_dir.exists()
    
    info = pd.read_csv(external_data_dir / 'info.csv').reset_index(drop=True)
    
    images = np.array([img.imread(images_dir / fname) for fname in tqdm(info['ImageFile'])])

    x = images[:, ::rescaling_factor, ::rescaling_factor, :-1].copy()
    y = info[['POS_X', 'POS_Y', 'POS_Z', 'Q_W', 'Q_X', 'Q_Y', 'Q_Z']].values.astype(np.float64)
    
    return x, y
