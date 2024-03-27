use async_tensorrt::*;

#[tokio::main]
async fn main() {
    // Set builder flags.
    let builder = Builder::new().await.unwrap();
    let mut builder = builder.with_default_optimization_profile().unwrap();
    let builder_config = builder.config().await
        .with_version_compability() // Needed to run with the lean runtime.
        .with_exclude_lean_runtime() // Don't include runtime with the engine.
        .with_fp16(); // Set precision.

    let network_definition =
        builder.network_definition(NetworkDefinitionCreationFlags::ExplicitBatchSize);

    // Load onnx file.
    let onnx_file_path = "input.onnx";
    let mut network_definition =
        Parser::parse_network_definition_from_file(network_definition, &onnx_file_path).unwrap();

    // Build the engine.
    let plan = builder
        .build_serialized_network(&mut network_definition, builder_config).await
        .unwrap();

    // Save serialized engine.
    let mut file = std::fs::File::create("build.engine").unwrap();
    use std::io::Write;
    file.write_all(plan.as_bytes()).unwrap();
}
