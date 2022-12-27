from __future__ import print_function

import os
import argparse
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torchvision.transforms as transforms
import common
from PIL import Image
from lib.utils.augmentations import letterbox_for_img
import torch
import ctypes
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
# ctypes.CDLL("./ScatterND.so", mode=ctypes.RTLD_GLOBAL)
EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
def build_engine(onnx_file_path, mode='fp16', verbose=False):
    """Build a TensorRT engine from an ONNX file."""
    
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # builder.max_workspace_size = 1 << 30
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        builder.max_batch_size = 1
        config.set_flag(trt.BuilderFlag.FP16)
        runtime = trt.Runtime(TRT_LOGGER)

        # if mode == 'fp16':
        #     # config.fp16_mode = True
        # else:
        #     builder.fp16_mode = False
        #builder.strict_type_constraints = True
        print("dooooooooooooooooooooooooooooooooooooooooo")
        # Parse model file
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("network.num_inputs", network.num_inputs)
        print("network.num_outputs", network.num_outputs)
        if trt.__version__[0] >= '7':
            # Reshape input to batch size 1
            shape = list(network.get_input(0).shape)
            shape[0] = 1
            network.get_input(0).shape = shape

        model_name = onnx_file_path[:-5]

        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" to enable verbose logging.)')
        # engine = builder.build_cuda_engine(network)
        plan = builder.build_serialized_network(network, config)
        # engine = runtime.deserialize_cuda_engine(plan)
        print('Completed creating engine.')
        return runtime.deserialize_cuda_engine(plan)


def load_img(path):
    img0 = cv2.imread(path, cv2.IMREAD_COLOR |
                      cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    h0, w0 = img0.shape[:2]

    img, ratio, pad = letterbox_for_img(img0.copy(), new_shape=640, auto=True)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img = np.ascontiguousarray(img)
    return img, img0, shapes
def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
def quantize(a):
    maxa,mina=np.max(a),np.min(a)
    c = (maxa- mina)/(255)
    d = np.round_((mina*255)/(maxa - mina))
    a = np.round_((1/c)*a-d)
    return a.astype('uint8'), c, d
from multiple import *
def main():
    # engine = build_engine(
    #     'onnx_export/det.onnx')
    # with open('trt8_tx2/det_16.trt', 'wb') as f:
    #     f.write(engine.serialize())
        
    engine = build_engine(
        'onnx_export/backbone.onnx')
    with open('trt_tx21/bb_16.trt', 'wb') as f:
        f.write(engine.serialize())
        
    engine = build_engine(
        'onnx_export/seg.onnx')
    with open('trt_tx21/seg_16.trt', 'wb') as f:
        f.write(engine.serialize())
    # model = backbone()
    # model.eval()
    # img, img_det, shapes = load_img(
    #         'inference/images/9aa94005-ff1d4c9a.jpg')
    # img = transform(img).to('cpu')
    # img = img.float()
    # if img.ndimension() == 3:
    #     img = img.unsqueeze(0)
    
    # outs = model(img)
    # outs = [ i.detach().cpu().numpy() for i in outs]
    # engine = load_engine('trt-cli/det_16.trt', False)
    # h_inputs, h_outputs, bindings, stream = common.allocate_buffers(engine)
    # with engine.create_execution_context() as context:
    #     h_inputs[0].host=outs[0]
    #     h_inputs[2].host=outs[1]
    #     h_inputs[2].host=outs[2]
    #     trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs, stream=stream)
    #     output = tensor(np.reshape(trt_outputs[3],(1, 15120, 6)))
    #     det_out = post_process_det(output,img,img_det)
    #     cv2.imwrite('trt_det.jpg',det_out)
if __name__ =='__main__':
    main()

#     img, img_det, shapes = load_img(
#         '/home/ceec/YOLOP/inference/images/8e1c1ab0-a8b92173.jpg')
#     img = transform(img).numpy()
    
#     img = np.expand_dims(img, axis=0)
#     # print(img.shape)
#     # img = img.float()
#     # if img.ndimension() == 3:
#     #     img = img.unsqueeze(0)

#     h_inputs[0].host = img

#     trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs, stream=stream)
#     for i,trt_output in enumerate(trt_outputs):
#         print(trt_output.shape)
#         print(trt_output.dtype)
#         print("min",np.min(trt_output))
#         print("max",np.max(trt_output))
#         np.save('tensor16b_{}.npy'.format(i), trt_output)
#         trt_output, c, d = quantize(trt_output)
#         # trt_output=trt_output.astype('uint8')
#         np.save('tensor8b_{}.npy'.format(i), trt_output)
#         trt_outputs[i] = c*(trt_output+d)
#     trt_outputs[0] = torch.from_numpy(trt_outputs[0].reshape(1, 256, 12, 20)).float()
#     trt_outputs[1] = torch.from_numpy(trt_outputs[1].reshape(1, 128, 24, 40)).float()
#     trt_outputs[2] = torch.from_numpy(trt_outputs[2].reshape(1, 256, 48, 80)).float()

# # print(trt_outputs.dtype)
# from multiple import *
# model_seg = seg_head()
# model_seg.eval()
# # da_seg_out = model_seg(trt_outputs[2])
# color_area = post_process_seg(da_seg_out, img, shapes)

# cv2.imwrite('test_trt2.jpg', color_area)
