pub(crate) const WINDOW_TITLE: &str = "Triad Visualizer";
pub(crate) const VISUALIZER_ENV_COUNT: usize = 128;

pub(crate) const DONE_REASON_COMPLETE: u32 = 1 << 0;
pub(crate) const DONE_REASON_GATE_COLLISION: u32 = 1 << 1;
pub(crate) const DONE_REASON_OBSTACLE_COLLISION: u32 = 1 << 2;
pub(crate) const DONE_REASON_FLOOR_COLLISION: u32 = 1 << 3;
pub(crate) const DONE_REASON_OUT_OF_BOUNDS: u32 = 1 << 4;
pub(crate) const DONE_REASON_STEP_LIMIT: u32 = 1 << 5;
pub(crate) const DONE_REASON_EXCESSIVE_TILT: u32 = 1 << 6;
