from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import image as img
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from myutils import log
from myutils.json_tools import save_json

cropped_size = 180


def get_random_crop(image: np.ndarray, cropped_size: int) -> np.ndarray:
    """Get a random square crop of the given image.
    """
    
    half_size = cropped_size // 2

    height, width, _ = image.shape
    center_x = np.random.randint(half_size, height - half_size)
    center_y = np.random.randint(half_size, width - half_size)
    
    return image[center_x - half_size : center_x + half_size]


def get_center_crop(image: np.ndarray, cropped_size: int) -> np.ndarray:
    """Get the center crop of the given image.
    """
    
    height, width, _ = image.shape
    top_left_x = (height - cropped_size) // 2
    top_left_y = (width - cropped_size) // 2
    
    return image[top_left_x:top_left_x+cropped_size, top_left_y:top_left_y+cropped_size]


log('Loading data.')

root_dir = Path(__file__).absolute().parent.parent

data_dir = root_dir / 'data'
data_dir.mkdir(exist_ok=True)

models_dir = root_dir / 'models'
models_dir.mkdir(exist_ok=True)

external_data_dir = Path('/home/ylytkin/Development/data/camera_relocalization_sample_dataset')
images_dir = external_data_dir / 'images'

info = pd.read_csv(external_data_dir / 'info.csv').reset_index(drop=True)

log('Loading images.')
images = np.array([img.imread(images_dir / fname) for fname in tqdm(info['ImageFile'])])

x = images[:, ::2, ::2, :-1].copy()
y = info[['POS_X', 'POS_Y', 'POS_Z', 'Q_W', 'Q_X', 'Q_Y', 'Q_Z']].values.astype(np.float64)

print(f'x.shape = {x.shape}, y.shape = {y.shape}')
log('Splitting data.')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

log('Augmenting train data.')

x_train_augmented = []
y_train_augmented = []

for x_, y_ in zip(x_train, y_train):
    for _ in range(30):
        x_train_augmented.append(get_random_crop(x_, cropped_size=cropped_size))
        y_train_augmented.append(y_)
        
x_train_augmented = np.array(x_train_augmented)
y_train_augmented = np.array(y_train_augmented)

print(f'x_train_augmented.shape = {x_train_augmented.shape}, y_train_augmented.shape = {y_train_augmented.shape}')
log('Center-cropping test data.')

x_test_centered = np.array([get_center_crop(image, cropped_size=cropped_size) for image in x_test])

print(f'x_test_centered.shape = {x_test_centered.shape}, y_test.shape = {y_test.shape}')


def loss_(y_true, y_pred, beta: float):
    """Squared error sum with promoting the quaternion part error by the `beta` arguement.
    """
    
    y_true_pos = y_true[:, :4]
    y_true_q = y_true[:, 4:]
    y_pred_pos = y_pred[:, :4]
    y_pred_q = y_pred[:, 4:]
    
    return (tf.reduce_sum(tf.square(y_true_pos - y_pred_pos), axis=1)
            + beta * tf.reduce_sum(tf.square(y_true_q - y_pred_q), axis=1))

log('Constructing tensorflow model.')

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=x_train_augmented[0].shape),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(y_train_augmented.shape[1]),
])
model.summary()

beta_options = list(range(150, 800, 150))
results = []
epochs = 10

log('Performing grid search.')
print(f'Options for `beta`: {beta_options}. Training for {epochs} epochs for each `beta`.')

for beta in tqdm(beta_options):
    log(f'beta = {beta}')
    
    def loss(y_true, y_pred):
        """Current loss. See `loss_` above.
        """
        
        return loss_(y_true, y_pred, beta=beta)

    model.compile(loss=loss, metrics=['mse'])
    training_history = model.fit(
        x_train_augmented, y_train_augmented,
        epochs=epochs,
        validation_data=[x_test_centered, y_test],
    )
    result = training_history.history
    
    y_pred = model.predict(x_test_centered)
    y_pred[:, 4:] /= ((y_pred[:, 4:] ** 2).sum(axis=1) ** 0.5).reshape(-1, 1)
    
    mse_score = tf.keras.metrics.mse(y_test, y_pred).numpy().mean()
    loss_score = loss(y_test, y_pred).numpy().mean()
    
    result['true_mse'] = mse_score
    result['true_loss'] = loss_score
    
    results.append(result)

results_fpath = data_dir / 'posenet_gridsearch_results.json'
    
log(f'Done. Saving results to: {results_fpath.as_posix()}.')
save_json(results, results_fpath)

log('Done.')
