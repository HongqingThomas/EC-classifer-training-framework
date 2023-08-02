import onnx
onnx_model = onnx.load("CNN.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("model incorrect")
else:
    print("model correct")

import onnxruntime
import cv2
import numpy as np
import torch

ort_session  = onnxruntime.InferenceSession("cnn.onnx")
# input_img = cv2.imread("./datasets/data/visible/0001.jpeg").astype(np.float32)
# input_img = np.transpose(input_img, (2, 0, 1))  # Change from HWC to CHW format
# input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
input_img = np.random.rand(16, 3,120,144).astype(np.float32)
ort_input = {'input': input_img}
ort_output = ort_session.run(['output'], ort_input)
print("output:", ort_output)