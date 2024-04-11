tensor_rt inference with small docker files + rust
----

Example where we create a tensor_rt engine file from a onnx file and
use it for inference using `rust` + `libnvinfer_lean.so`. Compressed docker container
ends up around 100mb.

```bash
# download TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz to ./downloads
docker build . -t tensor_rt

# copy a test input image (not included in repo) to ./test_input.jpg

# run inference:
docker run -it --rm --gpus all -v $(pwd):/mnt tensor_rt
# output:
#
# Num io tensors 2
# IO Tensor 0 name "input"
#   shape [1, 3, 360, 640]
#   mode Input
#   data type FLOAT
# IO Tensor 1 name "output"
#   shape [1, 1, 360, 640]
#   mode Output
#   data type INT32
# Done. Hits 127409 / 230400.

# inference results are written to ./output.png

# Save compressed docker file
docker save sha256:... | gzip > runner.tar.gz
ls -sh runner.tar.gz
# 102M runner.tar.gz
```
