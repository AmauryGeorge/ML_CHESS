import cv2
import typing
import numpy as np
import os

from mltu.configs import BaseModelConfigs

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import get_cer, get_wer
from decoder import ctc_decoder

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self,char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list
    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text,unsure = ctc_decoder(preds, self.char_list)

        return text[0],unsure

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    configs = BaseModelConfigs.load("torchModel/Models/BestModels/best/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    accum_cer = []
    accum_wer = []
    l=[]

    dataset, vocab, max_len = [], set(), 0

    dataset_path = "data"
    # for i in tqdm(range(6000, 10000)):
    #     img_path = os.path.join(dataset_path, f"{i}.png")
    #     label_path = os.path.join(dataset_path, f"{i}.txt")
        
    #     if not os.path.exists(img_path) or not os.path.exists(label_path):
    #         print(f"File not found: {img_path} or {label_path}")
    #         continue

    #     with open(label_path, 'r') as file:
    #         label = file.read().strip()
    #     l.append([img_path, label])
    """data_cr = pd.read_csv("test_data/prediciton.csv")
    ids_label = data_cr[["id", "prediction"]]
    tests = ids_label[1000:]
    for id in tqdm(tests["id"]):
        #try either png or jpe
        img_path = os.path.join("../test_data/images", f"{id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join("../test_data/images", f"{id}.jpe")
        label = ids_label[ids_label["id"] == id]["prediction"].values[0]
        l.append([img_path, label])"""
    
    data_cr = pd.read_csv("test_data/prediciton.csv")
    ids_label = data_cr[["id", "prediction"]]
    train_ids_label = ids_label[1000: ]
    for id in tqdm(train_ids_label["id"]):
        #try either png or jpe
        img_path = os.path.join("test_data/images", f"{id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join("test_data/images", f"{id}.jpe")
        label = ids_label[ids_label["id"] == id]["prediction"].values[0]
        dataset.append([img_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))
        l.append([img_path, label])


    l.append(["manual_test/img2.jpeg", "Qc9"])
    """l.append(["manual_test/img1.jpeg", "axb8"])
    l.append(["manual_test/img3.jpeg", "Kb7+"])
    l.append(["manual_test/img4.jpeg", "Rb8"])
    l.append(["manual_test/img5.jpeg", "Bb6"])"""
    for image_path, label in tqdm(l):
        image = cv2.imread(image_path)
        #reshape image to be 32x128
        
        prediction_text, unsure = model.predict(image)
        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)

        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}, unsures: {unsure} ")

        accum_cer.append(cer)
        accum_wer.append(wer)
        # resize by 4x
        """image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

    print(f"Average CER: {np.average(accum_cer)}")
    print(f"Average WER: {np.average(accum_wer)}")