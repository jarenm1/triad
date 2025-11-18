use std::env;
use std::path::PathBuf;

fn main() {
    let ply_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("goat.ply"));

    if let Err(err) = triad_window::run(&ply_path) {
        eprintln!("triad-window failed: {err}");
    }
}

