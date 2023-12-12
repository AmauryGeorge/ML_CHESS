import tensorflow as tf
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import train_model
from configs import ModelConfigs

import os
import tarfile
from tqdm import tqdm

# Updated dataset path
dataset_path = "data2"

# Initialize dataset, vocab, and max_len
dataset, vocab, max_len = [], set(), 0

# Load and preprocess the dataset
for i in tqdm(range(50000, 51499)):
    img_path = os.path.join(dataset_path, f"{i}.png")
    label_path = os.path.join(dataset_path, f"{i}.gt.txt")
    
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print(f"File not found: {img_path} or {label_path}")
        continue

    with open(label_path, 'r') as file:
        label = file.read().strip()

    dataset.append([img_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))
# Updated dataset path
dataset_path = "data"

# Load and preprocess the dataset
for i in tqdm(range(0, 1499)):
    img_path = os.path.join(dataset_path, f"{i}.png")
    label_path = os.path.join(dataset_path, f"{i}.gt.txt")
    
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print(f"File not found: {img_path} or {label_path}")
        continue

    with open(label_path, 'r') as file:
        label = file.read().strip()

    dataset.append([img_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

data_cr = pd.read_csv("test_data/prediciton.csv")
ids_label = data_cr[["id", "prediction"]]
train_ids_label = ids_label[0:1000]
for id in tqdm(train_ids_label["id"]):
    #try either png or jpe
    img_path = os.path.join("test_data/images", f"{id}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join("test_data/images", f"{id}.jpe")
    label = ids_label[ids_label["id"] == id]["prediction"].values[0]
    dataset.append([img_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split = 0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
    ]

# Creating TensorFlow model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
    dropout=0.15,
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
    loss=CTCloss(), 
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
)
model.summary(line_length=110)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))