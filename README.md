# What you need
1. Code Platform: Google Collaboration / Kaggle and VSCode
2. Programming Language: Python
3. Library: tensorflow, cv2, skearn, os, request, bs4, numpy

# Dataset
The dataset can be downloaded via:
1. Kaggle (https://www.kaggle.com/datasets/arifnuriman/indonesian-public-figure-faces)

# Steps for Training
1. Import data into the Code Platform
2. Resize the image into the desired size, we resize it to 224x224
3. Define the pre-trained model as the base model, we use a pre-trained model of VGG Face (Weights: https://www.kaggle.com/datasets/acharyarupak391/vggfaceweights)
4. Add flatten, Lambda layers as the top layer, Lambda used L2 Normalization method
5. Construct Siamese model with triplet loss to train encoder model
6. Define loss function and optimizer.
7. Train the model
8. After obtaining the desired accuracy, we can convert the model into a keras saved model
9. (Optional) Compress the model to tflite using tf.float16 option for quantization
```
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
```
For the complete code checkout the notebook on https://github.com/verifikasiin/siamese-face-recognition-model/blob/main/2%20Preprocess%20%26%20Train%20Model.ipynb

# Steps for Predicting
1. Load the tflite model using read byte and allocate tensor for the interpreter
  ```
   with open(model_path, 'rb') as f:
    model_tflite = f.read()
  interpreter = tf.lite.Interpreter(model_content=model_tflite)
  interpreter.allocate_tensors()
   ```
2. Read image then detect and crop face from image
3. Resize the face image to 224x224
4. Invoke the interpreter for the encoder model
5. Measure the distance between encoded value of two image

For the complete code on inference, checkout the python script https://github.com/verifikasiin/siamese-face-recognition-model/blob/main/5%20face_verification_lite.py 
