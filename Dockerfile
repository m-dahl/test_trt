# FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:21196d81f56b48dbee70494d5f10322e1a77cc47ffe202a3bf68eab81533c20f as builder

ARG TENSORRT_VERSION=8.6.1.6
ARG CUDA_USER_VERSION=12.0
ARG CUDNN_USER_VERSION=8.9
ARG OPERATING_SYSTEM=Linux

ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_CUDA="1"

# Install package dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         wget \
         curl \
         unzip \
         sudo \
         cmake \
         build-essential \
         libjpeg-dev \
         libpng-dev \
         language-pack-en \
         locales \
         locales-all \
         autoconf \
         automake \
         libtool \
         pkg-config \
         ca-certificates \
         git \
         && apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Copy TensorRT
COPY ./downloads/TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz /opt
RUN cd /opt && \
    tar -xzf TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz && \
    rm TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-${TENSORRT_VERSION}/lib
ENV PATH=$PATH:/opt/TensorRT-${TENSORRT_VERSION}/bin

# Install rust.
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Make links to tensor rt where async-tensorrt looks for them.
RUN mkdir /usr/local/tensorrt \
    && ln -s /opt/TensorRT-8.6.1.6/include /usr/local/tensorrt/include \
    && ln -s /opt/TensorRT-8.6.1.6/lib /usr/local/tensorrt/lib64

# Build rust example builder (generate engine file from ONNX file)
COPY ./example_builder /example_builder
RUN cd /example_builder && . ~/.cargo/env && cargo build --release # cargo run --release
# Now we only build the builder project, to run it we must run with the GPU!
# I have added the resulting engine file to the repo instead.

# Download and build dependencies
RUN mkdir /example_runner
COPY ./example_runner/Cargo.toml /example_runner
RUN mkdir /example_runner/src && echo "fn main() {}" > /example_runner/src/main.rs
RUN cd /example_runner && . ~/.cargo/env && cargo build --release
# Delete dummy src folder.
RUN rm -rf /example_runner/src

# Build rust example runner (runs inference using engine file produced before)
COPY ./example_runner/src /example_runner/src
RUN touch /example_runner/src/main.rs
RUN cd /example_runner && . ~/.cargo/env && cargo build --release

# STAGE2: create a slim(mer) image with the compiled runner binary
FROM nvcr.io/nvidia/cuda:12.1.1-base-ubuntu22.04@sha256:457a4076c56025f51217bff647ca631c7880ad3dbf546b03728ba98297ebbc22

# Full version. (comment out for slim)
# FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:21196d81f56b48dbee70494d5f10322e1a77cc47ffe202a3bf68eab81533c20f

# Copy lean tensor rt lib to runner docker
COPY --from=builder /opt/TensorRT-8.6.1.6/lib/libnvinfer_lean.so.8 /opt/TensorRT-8.6.1.6/lib/libnvinfer_lean.so.8

# Copy full tensor rt libs (comment out for slim)
# COPY --from=builder /opt/TensorRT-8.6.1.6/lib/libnvinfer.so.8 /opt/TensorRT-8.6.1.6/lib/libnvinfer.so.8
# COPY --from=builder /opt/TensorRT-8.6.1.6/lib/libnvonnxparser.so.8 /opt/TensorRT-8.6.1.6/lib/libnvonnxparser.so.8
# COPY --from=builder /opt/TensorRT-8.6.1.6/lib/libnvinfer_plugin.so.8 /opt/TensorRT-8.6.1.6/lib/libnvinfer_plugin.so.8

# Set tensor rt dylib path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-8.6.1.6/lib

# Copy the runner binary from the builder stage
WORKDIR /example_runner
COPY --from=builder /example_runner/target/release/example_runner example_runner

CMD ["./example_runner"]
