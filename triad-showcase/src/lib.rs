#[allow(dead_code)]
#[path = "main.rs"]
mod showcase_app;

pub use showcase_app::run_showcase_native;

#[cfg(target_arch = "wasm32")]
pub use showcase_app::run_showcase_web;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start_showcase(canvas_id: String) -> Result<(), String> {
    console_error_panic_hook::set_once();
    run_showcase_web(canvas_id).map_err(|err| err.to_string())
}
