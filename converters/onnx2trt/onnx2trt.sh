/usr/src/tensorrt/bin/trtexec \
--maxBatch=256 \
--onnx=densnet121.onnx \
--saveEngine=densnet121.trt
#--explicitBatch \
#--shapes=net_input:1x3x112x112 \
#--optShapes=net_input:2x112x112x3 \
#--minShapes=net_input:1x112x112x3 \
#--maxShapes=net_input:256x112x112x3 \
#--verbose \
#--onnx=densnet121.onnx  \
#--saveEngine=densnet121.trt \
