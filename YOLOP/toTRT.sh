#trtexec --onnx=onnx_export/backbone.onnx --saveEngine=cli_trt/bb.trt  --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --allowGPUFallback --workspace=1500
trtexec --onnx=det.onnx --saveEngine=cli_trt/det.trt  --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
#trtexec --onnx=onnx_export/seg.onnx --saveEngine=cli_trt/seg.trt  --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --allowGPUFallback --workspace=1500
