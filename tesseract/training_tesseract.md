# How we fined tuned tesseract for chess moves recognition

In this document, we will briefly explain the training process we went through to attempt fined tuning tesseract for chess moves recognition, in an effort of reproducability.

## Ressources

The main ressources used during this project were the tesseract organization's repositories:

- The main model's repo: https://github.com/tesseract-ocr/tesseract
- The one containing the best trained model for each language: https://github.com/tesseract-ocr/tessdata_best
- The tesstrain repository, used to train the model with make: https://github.com/tesseract-ocr/tesstrain

We also link the used tesseract documentation as well as a video tutorial describing the training process:

- https://tesseract-ocr.github.io/
- https://youtu.be/KE4xEzFGSU8

## How to train

1. Utilize an environment with 'make' (preferably on Ubuntu for Windows users; Mac users should have no issues) and install Tesseract OCR (sudo apt install tesseract-ocr)

2. Switch to root

3. Create a 'tesseract' folder and navigate inside.

4. Clone these two repositories:
   - https://github.com/tesseract-ocr/tesstrain.git
   - https://github.com/tesseract-ocr/tesseract.git
5. Within the Tesseract repository, clone the tessdata repository: https://github.com/tesseract-ocr/tessdata_best.git. Note: Change the name to 'tessdata' (git clone https://github.com/tesseract-ocr/tessdata_best.git tessdata). Be aware that there is already a 'tessdata' folder in the Tesseract repository; merge the data by copying the files from 'tesseract/tessdata' into the newly cloned repository.

6. Inside the tesstrain repository, create a 'data' folder. This folder will contain all model data and training data.

7) To add an initial model (using the 'eng' model of Tesseract here):
   - Visit https://github.com/tesseract-ocr/langdata, copy the folder of the desired model (e.g., 'eng'), and place it into our 'data' folder.
   - Go to https://github.com/tesseract-ocr/tessdata_best, download 'eng.traineddata,' and place it in 'tesstrain/data/eng.'
   - Clone https://github.com/tesseract-ocr/langdata.git inside 'tesstrain/data/'.
8) To train a model, execute the 'make' command in the tesstrain repository, providing the data inside 'tesstrain/data' and naming the folder 'MODEL_NAME-ground-truth.' Once created, copy the data into this folder (start with a subset initially to avoid extended processing time). Use the following command (inside tesstrain):

   ```
   TESSDATA_PREFIX=../tesseract/tessdata make training MODEL_NAME=MODEL_NAME START_MODEL= eng TESSDATA=../tesseract/tessdata MAX_ITERATIONS=100
   ```

   The english starting model and number of iterations is stated here as an example. The 'MAX_ITERATIONS' parameter can be changed to a higher number to improve accuracy. The 'START_MODEL' parameter can be changed to a different language model (e.g., 'fra').

9. Once the model is trained, you can run the following command in tesstrain to test an image:

   ```
   tesseract PATH_TO_FILE stdout --tessdata-dir data/ --psm 7 -l MODEL_NAME
   ```

   Replace 'PATH_TO_FILE' with the path to the image to be tested (e.g., data/MODEL_NAME-ground-truth/image_name.png).

Tips: Issues with outdated commands can be ecountered, if you do, update them (e.g., sudo apt-get install bc).
