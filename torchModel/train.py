import os
import tarfile
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd 

import torch
import torch.optim as optim
from torchsummaryX import summary

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from model import Network
from configs import ModelConfigs

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


configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

# Split the dataset into training and validation sets
train_dataProvider, test_dataProvider = data_provider.split(split = 0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
    ]

network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3)
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

# uncomment to print network summary, torchsummaryX package is required
#summary(network, torch.zeros((1, configs.height, configs.width, 3)))

# put on cuda device if available
network = network.cpu()

# create callbacks
earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.height, configs.width, 3), 
    verbose=1,
    metadata={"vocab": configs.vocab}
    )

# create model object that will handle training and testing of the network
model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])
model.fit(
    train_dataProvider, 
    test_dataProvider, 
    epochs=1000, 
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
    )

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))