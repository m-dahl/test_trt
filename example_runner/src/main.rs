use std::collections::HashMap;
use std::io::prelude::*;
use std::fs::{self, File};
use image::{self, Pixel, RgbaImage};
use std::path::{Path, PathBuf};
use std::time::{Instant, Duration};
use async_channel::{Receiver, Sender};
use clap::Parser;

const IMG_WIDTH: usize = 640;
const IMG_HEIGHT: usize = 360;

// Make a version of this that sources images from the redis server at tuve.
async fn read_data_dir<P: AsRef<Path>>(data_dir: P, input_tx: Sender<(String,
                                                                      RgbaImage,
                                                                      Vec<f32>)>) -> std::io::Result<()> {
    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();

        let is_file = fs::metadata(&path)?.is_file();
        let is_jpg = path.extension() == Some(std::ffi::OsStr::new("jpg"));

        if is_file && is_jpg {
            let img = image::open(&path).expect("File not found!");
            let rgba = img.into_rgba8();
            let tensor = image_to_tensor(&rgba,
                                         &[123.675, 116.28, 103.53],
                                         &[58.395, 57.12, 57.375]);

            let name = path.file_stem()
                .and_then(|f| f.to_str())
                .unwrap_or("no_filename").to_string();
            // println!("Prepared image: {}", name);
            if input_tx.send((name,rgba, tensor)).await.is_err() {
                println!("Sending failed");
            };
        }
    }
    Ok(())
}

// Make a version of this that save results to the redis server at tuve.
async fn write_data_dir(data_dir: &str, mask_rx: Receiver<(String,
                                                           RgbaImage,
                                                           Vec<i32>)>) -> std::io::Result<usize> {
    let mut count = 0;
    while let Ok((name, mut rgba, mask)) = mask_rx.recv().await {
        let path = PathBuf::from(data_dir);
        let mut filename = PathBuf::from(&name);
        filename.set_extension("png");
        let path = path.join(filename);
        let mut i = 0;

        for p in rgba.pixels_mut() {
            if mask[i] == 1 {
                p.blend(&Pixel::from_slice(&[0, 255, 0, 148]));
            }
            i+=1;
        }
        rgba.save(path).expect("Could not save image");
        // println!("Saved result {}", name);
        count += 1;
    }
    Ok(count)
}

/// Perform inference on the input data until the channel closes.
async fn inference_task(mut context: async_tensorrt::ExecutionContext<'_>,
                        input_rx: Receiver<(String, RgbaImage, Vec<f32>)>,
                        output_tx: Sender<(String, RgbaImage, Vec<i32>)>) {
    let stream = async_cuda::Stream::new().await.unwrap();

    let input_len: usize = 1 * 3 * IMG_WIDTH * IMG_HEIGHT;
    let num_input_bytes = input_len * std::mem::size_of::<f32>();
    let mut input_host_buffer = async_cuda::HostBuffer::<u8>::new(num_input_bytes).await;
    let input_device_bytes = async_cuda::DeviceBuffer::<u8>::new(num_input_bytes, &stream).await;

    let output_len: usize = 1 * 1 * IMG_WIDTH * IMG_HEIGHT;
    let num_output_bytes = output_len * std::mem::size_of::<i32>();
    let mut output_host_buffer = async_cuda::HostBuffer::<u8>::new(num_output_bytes).await;
    let output_device_bytes = async_cuda::DeviceBuffer::<u8>::new(num_output_bytes, &stream).await;

    let mut io_buffers = HashMap::from([
        ("output", output_device_bytes),
        ("input", input_device_bytes),
    ]);

    while let Ok((name,img,input_tensor)) = input_rx.recv().await {
        // let inf_start = Instant::now();
        if let Some(input) = io_buffers.get_mut("input") {
            let slice: &[u8] = unsafe { std::slice::from_raw_parts(input_tensor.as_ptr() as *const u8,
                                                                   num_input_bytes) };
            input_host_buffer.copy_from_slice(&slice);
            input.copy_from(&input_host_buffer, &stream).await.expect("Could not copy data");
        }
        let mut io_buffers_ref = io_buffers
            .iter_mut()
            .map(|(name, buffer)| (*name, buffer))
            .collect();
        context.enqueue(&mut io_buffers_ref, &stream).await.expect("Could not enqueue");

        io_buffers["output"].copy_to(&mut output_host_buffer, &stream).await.unwrap();
        let slice: &[i32] = unsafe {
            std::slice::from_raw_parts(output_host_buffer
                                       .inner()
                                       .as_internal()
                                       .as_ptr() as *const i32, output_len)
        };
        let output = slice.to_vec();

        // let inf_dur = inf_start.elapsed();
        // println!("Inference done in {}us", inf_dur.as_micros());
        if output_tx.send((name, img, output)).await.is_err() {
            println!("Could not send inference result");
        }
    }
}

/// Run benchmark. Returns images processed and time for inference
/// (not including loading/saving images).
async fn benchmark(model: &str, num_workers: usize) -> (usize, Duration) {
    // Use unbounded channels to pre-load all images.
    let (input_tx, input_rx) = async_channel::unbounded();
    let (output_tx, output_rx) = async_channel::unbounded();

    read_data_dir("/mnt/vt_data", input_tx).await.expect("Could not files");

    // Input channel now contains all images to be processed.
    let num_images = input_rx.len();

    // Load engine
    let mut f = File::open(model).expect("Could not load engine file");
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).unwrap();

    let mut runtime = async_tensorrt::Runtime::new().await;
    // Needs to be set if not kEXCLUDE_LEAN_RUNTIME is set when
    // building the engine file.
    runtime.set_engine_host_code_allowed(true);
    let engine = runtime.deserialize_engine(buffer.as_slice()).await.expect("Could not load engine");
    let contexts = async_tensorrt::ExecutionContext::from_engine_many(engine, num_workers).await
        .expect("Could create exection context");

    // Start timing clock
    let start = Instant::now();

    let mut inference_tasks = tokio::task::JoinSet::<usize>::new();
    for (i, context) in contexts.into_iter().enumerate() {
        let output_tx_clone = output_tx.clone();
        let input_rx_clone = input_rx.clone();
        {
            inference_tasks.spawn(async move {
                inference_task(context, input_rx_clone, output_tx_clone).await;
                return i;
            });
        }
    }

    // finish all inference jobs.
    while let Some(_res) = inference_tasks.join_next().await {
        // println!("Finished worker task: {:?}", res);
    }

    // Inference done.
    let duration = start.elapsed();

    // finish write stream.
    drop(output_tx);

    // save results (not really needed)
    write_data_dir("/mnt/output", output_rx).await.expect("Could not write files");

    // return results
    (num_images, duration)
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if args.bench {
        let models = &["/mnt/tensorrt_models/ampere_plus/trt_fp32/obstacles_640x360_fp32.engine",
                       "/mnt/tensorrt_models/ampere_plus/trt_fp16/obstacles_640x360_fp16.engine",
                       "/mnt/tensorrt_models/ampere_plus/trt_i8/obstacles_640x360_i8.engine" ];

        for model in models {
            println!("With model: {}", model);
            let (num_images, duration) = benchmark(model, args.workers).await;
            let fps = (num_images as f32) / duration.as_secs_f32();
            println!(" - Inference on {} images in {}ms", num_images, duration.as_millis());
            println!(" - Avg inference time: {}us (FPS: {})", duration.as_micros() / (num_images as u128), fps);
        }

        return;
    }

    let start = Instant::now();

    let (input_tx, input_rx) = async_channel::bounded(10);
    let (output_tx, output_rx) = async_channel::bounded(10);
    let input_jh = tokio::task::spawn(async {
        println!("Loading images start.");
        read_data_dir("/mnt/vt_data", input_tx).await.expect("Could not files");
        println!("Loading images done.");
    });

    let output_jh = tokio::task::spawn(async {
        println!("Saving images start.");
        let count = write_data_dir("/mnt/output", output_rx).await;
        println!("Saving images done.");
        count
    });

    // Load our engine file prepared by the builder.
    let mut f = File::open(&args.model_file).expect("Could not load engine file");
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).unwrap();

    let mut runtime = async_tensorrt::Runtime::new().await;
    // Needs to be set if not kEXCLUDE_LEAN_RUNTIME is set when
    // building the engine file.
    runtime.set_engine_host_code_allowed(true);
    let engine = runtime.deserialize_engine(buffer.as_slice()).await.expect("Could not load engine");

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

    let contexts = async_tensorrt::ExecutionContext::from_engine_many(engine, args.workers).await
        .expect("Could create exection context");

    let mut inference_tasks = tokio::task::JoinSet::<usize>::new();
    for (i, context) in contexts.into_iter().enumerate() {
        let output_tx_clone = output_tx.clone();
        let input_rx_clone = input_rx.clone();
        {
            inference_tasks.spawn(async move {
                inference_task(context, input_rx_clone, output_tx_clone).await;
                return i;
            });
        }
    }

    // finish all inference jobs.
    while let Some(res) = inference_tasks.join_next().await {
        println!("Finished worker task: {:?}", res);
    }

    // finish write stream.
    drop(output_tx);

    input_jh.await.unwrap();
    let num_images = output_jh.await.unwrap().unwrap();

    let duration = start.elapsed();
    println!("Processed {} images in {}ms (including disk io)", num_images, duration.as_millis());
    println!("Avg time per image: {}us", duration.as_micros() / (num_images as u128));
}

/// Preprocess the image.
///
/// Perform per channel normalization as well as separate the
/// channels (go from h * w * 3 to 3 * h * w)
fn image_to_tensor(image: &RgbaImage,
                   rgb_mean: &[f32; 3],
                   rgb_stddev: &[f32; 3]) -> Vec<f32> {
    assert_eq!(image.width(), IMG_WIDTH as u32);
    assert_eq!(image.height(), IMG_HEIGHT as u32);
    let wh = (image.width() * image.height()) as usize;
    let mut tensor = vec![0.0f32; 3 * wh];
    let mut i = 0usize;
    for p in image.pixels() {
        tensor[i] = ((p.0[0] as f32) - rgb_mean[0]) / rgb_stddev[0];
        tensor[i + wh] = ((p.0[1] as f32) - rgb_mean[1]) / rgb_stddev[1];
        tensor[i + 2 * wh] = ((p.0[2] as f32) - rgb_mean[2]) / rgb_stddev[2];
        i+=1;
    }
    tensor
}


/// GPSS inference in rust.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Model name
    #[arg(short, long, default_value_t = String::from("/mnt/tensorrt_models/ampere_plus/trt_fp16/obstacles_640x360_fp16.engine"))]
    model_file: String,

    /// Benchmark mode
    #[arg(short, long, default_value_t = false)]
    bench: bool,

    /// GPU Workers
    #[arg(short, long, default_value_t = 3)]
    workers: usize,
}
