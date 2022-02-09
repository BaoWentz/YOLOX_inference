cd /home/ecnu-lzw/bwz/ocr-gy/YOLOX_inference

python3 onnx_inference.py \
-m /home/ecnu-lzw/bwz/ocr-gy/YOLOX_inference/onnx_models/yolox_s2023.onnx \
-i /home/ecnu-lzw/bwz/ocr-gy/steelDatasets/datasets_img/2019_2019-10-26103539.jpg \
-o /home/ecnu-lzw/bwz/ocr-gy/YOLOX_inference/onnx_outputs \
-s 0.3 --input_shape 576,768

# /home/ecnu-lzw/bwz/ocr-gy/YOLOX_inference/inferenceONNX.sh
