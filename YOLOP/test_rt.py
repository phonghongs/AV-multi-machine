import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from lib.utils.augmentations import letterbox_for_img
import torchvision.transforms as transforms

def load_img(path):
    img0 = cv2.imread(path, cv2.IMREAD_COLOR |
                      cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    h0, w0 = img0.shape[:2]

    img, ratio, pad = letterbox_for_img(img0.copy(), new_shape=640, auto=True)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img = np.ascontiguousarray(img)
    return img, img0, shapes

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

def main():
    f = open("/home/YOLOP/tr_export/bb.trt", "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()


    # need to set input and output precisions to FP16 to fully enable it
    output = np.empty([1, 1000], dtype = np.float16) 

    # allocate device memory
    d_input = cuda.mem_alloc(1 * 1574560)
    d_output = cuda.mem_alloc(1 * 4769440)
    #(1474560,4669440)
    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    img, img_det, shapes = load_img(
                '/home/YOLOP/inference/images/adb4871d-4d063244.jpg')
    img = transform(img)
    img = np.float16(img)
    img = np.expand_dims(np.array(img, dtype=np.float16), axis=0)
    print(img.shape)
    
    outs= predict(img,d_input,d_output,stream,bindings,context,output)
    outs= predict(img,d_input,d_output,stream,bindings,context,output)
    #size of 1 image 1474560
def predict(batch,d_input,d_output,stream,bindings,context,output): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output
if __name__ =="__main__":
    main()
