from lib.utils import plot_one_box, show_seg_result
# import torch_tensorrt
import random
import numpy as np
from lib.core.general import non_max_suppression, scale_coords
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
# import torch_tensorrt

# from torch2trt import tensorrt_converter,torch2trt
sys.path.append(os.getcwd())
# sys.path.append("lib/models")
# sys.path.append("lib/utils")
# sys.path.append("/workspace/wh/projects/DaChuang")
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect


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
    # _, _, height, width = img.shape
    shapes = ((720, 1280), ((0.5333333333333333, 0.5), (0.0, 12.0)))
    height, width = 384, 640
    # h, w, _ = img_det.shape
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    ratio = shapes[1][0][1]

    da_predict = da_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(
        da_predict, scale_factor=int(1/ratio), mode='area')
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


full = False
if __name__ == "__main__":
    # inference full model
    if full:
        img, img_det, shapes = load_img(
            '/hinference/images/0ace96c3-48481887.jpg')
        img = transform(img).to('cuda')
        img = img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        logger, _, _ = create_logger(
            cfg, cfg.LOG_DIR, 'demo')
        device = select_device(logger, 'cpu')
        model = get_net(cfg)
        checkpoint = torch.load('weights/End-to-end.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        det_out, da_seg_out, ll_seg_out = model(img)

        for det in det_out:
            print(type(det))
        color_area = post_process_seg(da_seg_out)
        # cv2.imwrite("testttttttttttt2.jpg", color_area)
    else:
        model = backbone()
        model.eval()
        model_seg = seg_head()
        model_seg.eval()
        model_det = det_head()
        model_det.eval()

        img, img_det, shapes = load_img(
            'inference/images/adb4871d-4d063244.jpg')
        img = transform(img).to('cpu')
        img = img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # inference backbone+neck->input: img, output: 3 tensor [10,14,16]
        # img = img.to('cuda')
        
        st = time.time()
        outs = model(img)
        # for i,out in enumerate(outs):
            
            # print(out.shape)

        # outs = [i.float() for i in outs]
        
    #     input=[
    # torch_tensorrt.Input((1, 3, 60, 60)), # Static NCHW input shape for input #1
    # ]
    #     trt_model = torch_tensorrt.compile(model,inputs=)
       
        # print(count_parameters(model_seg))
        #inference segmentation drive able head->input: tensor 16, output: drive able mask
        da_seg_out=model_seg(outs[2])
        print(da_seg_out.shape)

        da_seg_out = da_seg_out.detach().numpy()
        print(np.shape(da_seg_out))
        # color_area=post_process_seg(da_seg_out)
        # #inference detection head->input: 3 tensor [10,14,16], output: object detection
        # det_out = model_det(outs)
       
        # img_det=post_process_det(det_out)
        # et = time.time()
        # torch_fps = 1/(et-st)
        # cv2.imwrite("color_area.jpg", color_area)
        # cv2.imwrite("det_out.jpg", img_det)
        # img = img.half()
        # img = img.to('cuda')
        # print(type(outs))
    #     tr_model = torch_tensorrt.compile(model,
    #     inputs= [torch_tensorrt.Input((1, 3, 60, 60))],
    # enabled_precisions= { torch_tensorrt.dtype.half})
        
        # torch2trt
        
        # tr_model_seg = torch2trt(model_seg,[outs[2]])
        # tr_model_det = torch2trt(model_det,outs)
        # st = time.time()
        # tr_model_out = tr_model(img)
        # tr_model_seg_out = tr_model_seg(tr_model_out[2])
        # tr_model_det_out = tr_model_det(tr_model_out)
        # color_area=post_process_seg(tr_model_seg_out)
        # img_det=post_process_det(det_out)
        # et = time.time()
        # rt_fps = 1/(et-st)
        # print('rt fps: ',rt_fps)
        # print('torch fps: ',torch_fps)

        # onnx
        # torch.onnx.export(model,img,'./onnx_export/backbone.onnx',export_params=True,
        # input_names=['input'],output_names=['outs'],verbose=False,opset_version=12)

        # torch.onnx.export(model_seg,outs[2],'./onnx_export/seg.onnx',export_params=True,
        # input_names=['input'],output_names=['output'],verbose=False,opset_version=12)
        # for k, m in model_det.named_modules():
        #     if isinstance(m, Detect):
        #         # m.inplace = inplace
        #         # m.onnx_dynamic = dynamic
        #         m.export = True
        # import onnx

        # torch.onnx.export(model_det,args=(outs[0],outs[1],outs[2]),f='./onnx_export/det.onnx',export_params=True,
        # input_names=['inputs0','inputs1','inputs2'],output_names=['output'],verbose=False,opset_version=12)
        # torch.onnx.export(
        #     model_det,  # --dynamic only compatible with cpu
        #     outs,
        #     './onnx_export/det.onnx',
        #     verbose=False,
        #     opset_version=12,
        #     training=torch.onnx.TrainingMode.EVAL,
        #     do_constant_folding=True,
        #     input_names=['in1','in2','in3'],
        #     output_names=['output'],
        #     dynamic_axes=None)
        # model_onnx = onnx.load('./onnx_export/det.onnx')  # load onnx model
        # onnx.checker.check_model(model_onnx)  # check onnx model

        # import onnxsim

        # # LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
        # model_onnx, check = onnxsim.simplify(model_onnx)
        # onnx.save(model_onnx, './onnx_export/det.onnx')

