mod constants;
mod manager;
mod render;
mod replay;
mod ui;

use std::error::Error;
use std::sync::{Arc, Mutex};

use glam::Vec3;
use tracing::info;
use triad_window::{CameraPose, WindowConfig, run_with_renderer_config};

use crate::constants::WINDOW_TITLE;
use crate::manager::VisualizerManager;
use crate::ui::{UiState, draw_visualizer_ui};

fn main() -> Result<(), Box<dyn Error>> {
    init_logging();

    let ui_state = Arc::new(Mutex::new(UiState::default()));
    let ui_state_for_controls = Arc::clone(&ui_state);
    let ui_state_for_manager = Arc::clone(&ui_state);

    run_with_renderer_config(
        WINDOW_TITLE,
        WindowConfig::default(),
        move |controls| {
            controls.request_reset(CameraPose::new(
                Vec3::new(8.5, 5.5, 8.5),
                Vec3::new(0.0, 0.75, 0.0),
            ));

            controls.on_ui(move |ctx| {
                let mut ui = ui_state_for_controls.lock().expect("ui state poisoned");
                draw_visualizer_ui(ctx, &mut ui);
            });
        },
        move |renderer, registry, surface_format, _width, _height| {
            let manager =
                VisualizerManager::new(renderer, registry, surface_format, ui_state_for_manager)?;
            Ok(Box::new(manager))
        },
    )
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "triad_visualizer=info,triad_window=info".into()),
        )
        .with_target(false)
        .compact()
        .try_init();
    info!("starting visualizer");
}
