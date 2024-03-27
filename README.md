tensor_rt inference with small docker files + rust
----

Example where we create a tensor_rt engine file from a onnx file and
use it for inference using `rust` + `libnvinfer_lean.so`. Compressed docker container
ends up around 100mb.

```bash
# download TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz to ./downloads
docker build . -t tensor_rt
# Engine files and gpss input file is not included in this repo.
# (So example is not currently runnable)
# Save compressed docker file
docker save sha256:... | gzip > runner.tar.gz
ls -sh runner.tar.gz
# 102M runner.tar.gz
```
