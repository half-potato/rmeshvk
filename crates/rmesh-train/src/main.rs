//! rmesh-train: Differentiable tetrahedral radiance mesh trainer.
//!
//! Usage:
//!   rmesh-train <input.rmesh> [--epochs N] [--lr LR] [--width W] [--height H]

use anyhow::{Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rmesh-train <input.rmesh> [--epochs N] [--lr LR]");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[1]);
    log::info!("Loading scene from: {}", input_path.display());

    // Parse optional flags
    let mut config = rmesh_train::TrainConfig::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--epochs" => {
                config.epochs = args[i + 1].parse().context("Invalid --epochs")?;
                i += 2;
            }
            "--lr" => {
                let lr: f32 = args[i + 1].parse().context("Invalid --lr")?;
                config.lr_sh = lr;
                i += 2;
            }
            "--width" => {
                config.render_width = args[i + 1].parse().context("Invalid --width")?;
                i += 2;
            }
            "--height" => {
                config.render_height = args[i + 1].parse().context("Invalid --height")?;
                i += 2;
            }
            "--l2" => {
                config.loss_type = 1;
                i += 1;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Load scene
    let file_data = std::fs::read(&input_path)
        .with_context(|| format!("Failed to read {}", input_path.display()))?;
    let (scene, sh) = rmesh_data::load_rmesh(&file_data)
        .or_else(|_| rmesh_data::load_rmesh_raw(&file_data))
        .context("Failed to parse scene file")?;

    log::info!(
        "Scene: {} vertices, {} tets, SH degree {}",
        scene.vertex_count,
        scene.tet_count,
        sh.degree
    );

    // Create wgpu device (headless — no window)
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        log::info!("GPU: {:?}", adapter.get_info().name);

        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rmesh train"),
                    required_features: wgpu::Features::SUBGROUP,
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
            )
            .await
            .expect("Failed to create device")
    });

    // TODO: Load training views (camera poses + ground truth images)
    let views: Vec<rmesh_train::TrainingView> = vec![];

    if views.is_empty() {
        log::warn!("No training views provided. Run with a dataset directory.");
        return Ok(());
    }

    rmesh_train::train(&device, &queue, &scene, &sh, &views, &config)?;

    log::info!("Training complete.");
    Ok(())
}
