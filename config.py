
model = './onnx_models/yolox_s2023.onnx'  # the path of onnx model
output_dir = './onnx_outputs'  # save the detect image
score_thr = 0.3  # score thresh when nms
input_shape = (576, 768)  # h, w input shape for images to feed to the model
vis_out = True  # whether render the image
