use std::collections::HashMap;
use std::io::prelude::*;
use std::fs::File;
use image::{self, Pixel};

#[tokio::main]
async fn main() {
    // Load our engine file prepared by the builder.
    let mut f = File::open("/mnt/obstacles_640x360_i8.engine").unwrap();
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).unwrap();

    let mut runtime = async_tensorrt::Runtime::new().await;
    // Needs to be set if not kEXCLUDE_LEAN_RUNTIME is set when
    // building the engine file.
    runtime.set_engine_host_code_allowed(true);
    let mut engine = runtime.deserialize_engine(buffer.as_slice()).await.expect("Could not load engine");

    let num_io_tensors = engine.num_io_tensors();
    println!("Num io tensors {}", num_io_tensors);

    for i in 0..num_io_tensors {
        let name = engine.io_tensor_name(i);
        println!("IO Tensor {} name \"{}\"", i, name);

        let shape = engine.tensor_shape(&name);
        println!("  shape {:?}", shape);

        let io_mode = engine.tensor_io_mode(&name);
        println!("  mode {:?}", io_mode);

        let data_type = engine.tensor_data_type(&name);
        println!("  data type {:?}", data_type);
    }

    // Load image.
    let img = image::open("/mnt/test_input.jpg").expect("File not found!");
    let mut rgb = img.clone().into_rgba8();

    let input_len: usize = 360 * 640 * 3;
    let mut input_buf = vec![0f32; input_len];

    // From MMSegment pipeline.json:
    // Pre-processing:
    // "type": "Normalize",
    // "mean": [
    //     123.675,
    //     116.28,
    //     103.53
    // ],
    // "std": [
    //     58.395,
    //     57.12,
    //     57.375
    // ],
    // "to_rgb": true

    // Separate channels (i.e. Input shape: [1, 3, 360, 640])
    let mut i = 0usize;
    let wh = (rgb.width() * rgb.height()) as usize;
    for y in 0 .. rgb.height() {
        for x in 0 .. rgb.width() {
            let p = rgb.get_pixel(x,y);
            input_buf[i] = ((p.0[2] as f32) - 123.675) / 58.395;
            input_buf[i + wh] = ((p.0[1] as f32) - 116.28) / 57.12;
            input_buf[i + 2 * wh] = ((p.0[0] as f32) - 103.53) / 57.375;
            i+=1;
        }
    }

    let output_len: usize = 1 * 1 * 360 * 640;
    let output = vec![0i32; output_len];

    let stream = async_cuda::Stream::new().await.unwrap();
    let mut context = async_tensorrt::ExecutionContext::new(&mut engine).await
        .expect("Could create exection context");

    let mut io_buffers = HashMap::from([
        ("input", to_device_bytes(&input_buf, &stream).await),
        ("output", to_device_bytes(&output, &stream).await),
    ]);
    let mut io_buffers_ref = io_buffers
        .iter_mut()
        .map(|(name, buffer)| (*name, buffer))
        .collect();
    context.enqueue(&mut io_buffers_ref, &stream).await.unwrap();

    // Get output
    let output: Vec<i32> = from_device_bytes(&io_buffers["output"], &stream).await;

    // Overlay result on input image and save it.
    let mut i = 0;
    let mut hits = 0;

    for y in 0 .. rgb.height() {
        for x in 0 .. rgb.width() {
            let p = rgb.get_pixel_mut(x,y);
            if output[i] == 1 {
                let green_blend = Pixel::from_slice(&[0, 255, 0, 148]);
                p.blend(&green_blend);
                hits += 1;
            }
            i+=1;
        }
    }
    rgb.save("/mnt/output.png").expect("Could not save image");

    println!("Done. Hits {} / {}.", hits, output.len());
}


// Just using the below functions to erase the type since
// the hashmap input for enqueue is typed and we have different
// output and input types.
pub async fn to_device_bytes<T: Copy>(slice: &[T], stream: &async_cuda::Stream) -> async_cuda::DeviceBuffer<u8> {
    let num_bytes = slice.len() * std::mem::size_of::<T>();
    let byte_slice: &[u8] = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8,
                                                                num_bytes) };

    let device_buffer = async_cuda::DeviceBuffer::from_slice(byte_slice, stream).await
        .expect("Could not create device buffer");
    return device_buffer;
}

pub async fn from_device_bytes<T: Copy>(device_buffer: &async_cuda::DeviceBuffer<u8>,
                                        stream: &async_cuda::Stream) -> Vec<T> {
    let num_bytes = device_buffer.num_elements();
    let num_elements = num_bytes / std::mem::size_of::<T>();
    let mut host_buffer: async_cuda::HostBuffer<u8> = async_cuda::HostBuffer::new(num_bytes).await;
    device_buffer.copy_to(&mut host_buffer, &stream).await.unwrap();
    let slice: &[T] = unsafe { std::slice::from_raw_parts(host_buffer.inner().as_internal().as_ptr() as *const T, num_elements) };
    return slice.to_vec();
}
