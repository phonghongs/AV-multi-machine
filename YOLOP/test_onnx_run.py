from lib.utils import plot_one_box, show_seg_result
# import torch_tensorrt
import random
import numpy as np
from lib.core.general import non_max_suppression, output_to_target, scale_coords
from lib.models import get_net
from lib.config import cfg
from lib.utils import initialize_weights
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
import cv2
import torch
from torch import tensor
import torch.nn as nn
import math
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.utils.augmentations import letterbox_for_img
import torchvision.transforms as transforms
import sys
import os
import time

import tensorrt as trt
# from torch2trt import tensorrt_converter,torch2trt
sys.path.append(os.getcwd())
# sys.path.append("lib/models")
# sys.path.append("lib/utils")
# sys.path.append("/workspace/wh/projects/DaChuang")
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect

# import onnxruntime
class det_head(nn.Module):
    def __init__(self):
        super(det_head, self).__init__()
        device = 'cpu'
        self.model = get_net(cfg)
        checkpoint = torch.load('weights/End-to-end.pth', map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, input_):
        # out = []
        in_detect = []
        for i, block in enumerate(self.model.model):
            if i < 17:
                continue
            if i == 17:
                x =input_[2]
            elif i == 24:
                x = in_detect
            if isinstance(block.from_, list):
                if list(block.from_) == [-1, 14]:
                    x = [x, input_[1]]
                if list(block.from_) == [-1, 10]:
                    x = [x, input_[0]]
            x = block(x)
            if i == self.model.detector_index:
                det_out = x
                break
            if i in [17, 20, 23]:
                in_detect.append(x)
        return det_out


class seg_head(nn.Module):
    def __init__(self):
        super(seg_head, self).__init__()
        device = 'cpu'
        self.model = get_net(cfg)
        checkpoint = torch.load('weights/End-to-end.pth', map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        self.save = self.model.save
        layers = []
        for i, block in enumerate(self.model.model):
            if i < 25:
                continue
            layers.append(block)
            if i == 33:
                break
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        m = nn.Sigmoid()
        x = self.model(x)
        out = m(x)
        return out


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        device = 'cpu'
        self.model = get_net(cfg)
        checkpoint = torch.load('weights/End-to-end.pth', map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        self.save = self.model.save
        layers = []
        for i, block in enumerate(self.model.model):
            layers.append(block)
            if i == 16:
                break
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        cache = []
        out = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [
                    x if j == -1 else cache[j] for j in block.from_]  # calculate concat detect
            x = block(x)
            if i in [10, 14, 16]:  # save driving area segment result
                out.append(x)
            cache.append(x if block.index in self.save else None)
        return out


def load_img(path):
    img0 = cv2.imread(path, cv2.IMREAD_COLOR |
                      cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    h0, w0 = img0.shape[:2]

    img, ratio, pad = letterbox_for_img(img0.copy(), new_shape=640, auto=True)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img = np.ascontiguousarray(img)
    return img, img0, shapes


def post_process_seg(da_seg_out):
    _, _, height, width = img.shape
    h, w, _ = img_det.shape
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    ratio = shapes[1][0][1]

    da_predict = da_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(
        da_predict, scale_factor=int(1/ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    color_area = np.zeros(
        (da_seg_mask.shape[0], da_seg_mask.shape[1], 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    # cv2.imwrite("testttttttttttt.jpg", color_area)
    return color_area


def post_process_det(det_out):
    inf_out, _ = det_out
    det_pred = non_max_suppression(
        inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    det = det_pred[0]
    names = model_det.model.module.names if hasattr(
        model_det.model, 'module') else model_det.model.names
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]
    if len(det):
        det[:, :4] = scale_coords(
            img.shape[2:], det[:, :4], img_det.shape).round()
        for *xyxy, conf, cls in reversed(det):
            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img_det, label=label_det_pred,
                         color=colors[int(cls)], line_thickness=2)
    # cv2.imwrite("testttttttttttt3.jpg", img_det)
    return img_det


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
if __name__ == "__main__":
    model = backbone()
    model.eval()
    model_seg = seg_head()
    model_seg.eval()
    model_det = det_head()
    model_det.eval()
    img, img_det, shapes = load_img(
            '/home/ceec/YOLOP/inference/images/7dd9ef45-f197db95.jpg')
    img = transform(img).to('cpu')
    img = img.float()
    if img.ndimension() == 3:
            img = img.unsqueeze(0)
    # bb_session = onnxruntime.InferenceSession("onnx_export/backbone.onnx",providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])
    # det_session = onnxruntime.InferenceSession("onnx_export/det.onnx",providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])
    # seg_session = onnxruntime.InferenceSession("onnx_export/seg.onnx",providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])


    # bb_inputs = {bb_session.get_inputs()[0].name: to_numpy(img)}
    # out_bb = bb_session.run(None,bb_inputs)
    bb_torch_out = model(img)
    out = to_numpy(bb_torch_out)
    print(bb_torch_out.nbytes)
    # seg_inputs = {seg_session.get_inputs()[0].name: out_bb[2]}
    # seg_out = seg_session.run(None,seg_inputs)
    # seg_torch_out = model_seg(bb_torch_out[2])
    
    # #dynamic 
    # tar_shape = np.shape(seg_out)
    # seg_out = np.reshape(seg_out,(1, 2, tar_shape[3], tar_shape[4]))
    # color_area=post_process_seg(tensor(seg_out))
    # cv2.imwrite("onnx_color_area.jpg", color_area)

    # # det_inputs = {seg_session.get_inputs()[0].name:out_bb}
    # det_out = det_session.run(None,{'out_bb0':out_bb[0],'out_bb1':out_bb[1],
    # 'out_bb2':out_bb[2]})
    # det_torch = model_det(bb_torch_out)
    # # print(type(det_torch[1][0]))
    # # print('----------------------------------')
    # print(type(det_out[1]))
    # det_out = (tensor(det_out[0]),[tensor(det_out[1]),tensor(det_out[2]),tensor(det_out[3])])
    # img_det=post_process_det(det_out)
    # cv2.imwrite("onnx_det_out.jpg", img_det)


    