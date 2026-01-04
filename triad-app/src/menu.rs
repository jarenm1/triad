//! Structured egui menu system for the application.

use crate::layers::LayerMode;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc;
use triad_window::egui::{Context, TopBottomPanel};

/// Menu actions that can be triggered from the menu.
#[derive(Debug, Clone)]
pub enum MenuAction {
    /// Import a PLY file from the selected path.
    ImportPly(PathBuf),
    /// Connect to a camera device by index.
    ConnectCamera(u32),
    /// Disconnect from the current camera.
    DisconnectCamera,
}

/// Menu builder for creating structured egui menus.
pub struct MenuBuilder {
    show_file_menu: bool,
    show_view_menu: bool,
    show_help_menu: bool,
    mode_signal: Option<Arc<AtomicU8>>,
    action_sender: Option<mpsc::Sender<MenuAction>>,
}

impl MenuBuilder {
    /// Create a new menu builder.
    pub fn new() -> Self {
        Self {
            show_file_menu: true,
            show_view_menu: true,
            show_help_menu: true,
            mode_signal: None,
            action_sender: None,
        }
    }

    /// Enable or disable the file menu.
    pub fn with_file_menu(mut self, show: bool) -> Self {
        self.show_file_menu = show;
        self
    }

    /// Enable or disable the view menu.
    pub fn with_view_menu(mut self, show: bool) -> Self {
        self.show_view_menu = show;
        self
    }

    /// Enable or disable the help menu.
    pub fn with_help_menu(mut self, show: bool) -> Self {
        self.show_help_menu = show;
        self
    }

    /// Set the mode signal for mode switching.
    pub fn with_mode_signal(mut self, signal: Arc<AtomicU8>) -> Self {
        self.mode_signal = Some(signal);
        self
    }

    /// Set the action sender for menu actions (PLY import, camera connection).
    pub fn with_action_sender(mut self, sender: mpsc::Sender<MenuAction>) -> Self {
        self.action_sender = Some(sender);
        self
    }

    /// Build the menu function.
    pub fn build(self) -> impl FnMut(&Context) + Send + 'static {
        let show_file = self.show_file_menu;
        let show_view = self.show_view_menu;
        let show_help = self.show_help_menu;
        let mode_signal = self.mode_signal;
        let action_sender = self.action_sender;

        move |ctx: &Context| {
            // Top menu bar
            TopBottomPanel::top("menu_bar").show(ctx, |ui| {
                triad_window::egui::MenuBar::new().ui(ui, |ui: &mut triad_window::egui::Ui| {
                    if show_file {
                        ui.menu_button("File", |ui: &mut triad_window::egui::Ui| {
                            // PLY Import
                            if ui.button("Import PLY...").clicked() {
                                if let Some(ref sender) = action_sender {
                                    // Spawn file dialog in a separate thread to avoid blocking
                                    let sender_clone = sender.clone();
                                    std::thread::spawn(move || {
                                        if let Some(path) = rfd::FileDialog::new()
                                            .add_filter("PLY Files", &["ply"])
                                            .add_filter("All Files", &["*"])
                                            .pick_file()
                                        {
                                            let _ = sender_clone.send(MenuAction::ImportPly(path));
                                        }
                                    });
                                }
                                ui.close();
                            }

                            ui.separator();

                            if ui.button("Exit").clicked() {
                                // TODO: Implement exit
                                ui.close();
                            }
                        });
                    }

                    if show_view {
                        ui.menu_button("View", |ui: &mut triad_window::egui::Ui| {
                            if let Some(ref signal) = mode_signal {
                                let current_mode = read_mode(signal);

                                // Create a dropdown for layer selection
                                ui.label("View Layer:");

                                let mut selected_mode = current_mode;
                                let mut changed = false;

                                for mode in LayerMode::all() {
                                    let is_selected = *mode == current_mode;
                                    if ui.selectable_label(is_selected, mode.to_string()).clicked()
                                    {
                                        selected_mode = *mode;
                                        changed = true;
                                    }
                                }

                                // Update signal if mode changed
                                if changed {
                                    write_mode(signal, selected_mode);
                                }

                                ui.separator();

                                // Also show quick access buttons
                                ui.label("Quick Switch:");
                                ui.horizontal(|ui| {
                                    if ui.button("Points").clicked() {
                                        write_mode(signal, LayerMode::Points);
                                    }
                                    if ui.button("Gaussians").clicked() {
                                        write_mode(signal, LayerMode::Gaussians);
                                    }
                                    if ui.button("Triangles").clicked() {
                                        write_mode(signal, LayerMode::Triangles);
                                    }
                                });
                            }
                        });
                    }

                    // Camera menu
                    ui.menu_button("Camera", |ui: &mut triad_window::egui::Ui| {
                        if let Some(ref sender) = action_sender {
                            // List available camera devices
                            ui.label("Available Devices:");

                            // Try to list devices (this might fail, so we handle errors gracefully)
                            match list_camera_devices() {
                                Ok(devices) => {
                                    if devices.is_empty() {
                                        ui.label("No cameras found");
                                    } else {
                                        for (index, device_name) in devices.iter().enumerate() {
                                            if ui
                                                .button(format!("Connect: {}", device_name))
                                                .clicked()
                                            {
                                                let _ = sender
                                                    .send(MenuAction::ConnectCamera(index as u32));
                                                ui.close();
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    ui.label(format!("Error listing cameras: {}", e));
                                }
                            }

                            ui.separator();

                            if ui.button("Disconnect Camera").clicked() {
                                let _ = sender.send(MenuAction::DisconnectCamera);
                                ui.close();
                            }
                        } else {
                            ui.label("Camera actions not available");
                        }
                    });

                    if show_help {
                        ui.menu_button("Help", |ui: &mut triad_window::egui::Ui| {
                            if ui.button("About").clicked() {
                                // TODO: Show about dialog
                                ui.close();
                            }
                        });
                    }
                });
            });
        }
    }
}

impl Default for MenuBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create a default menu with mode switching.
pub fn create_default_menu(mode_signal: Arc<AtomicU8>) -> impl FnMut(&Context) + Send + 'static {
    MenuBuilder::new().with_mode_signal(mode_signal).build()
}

/// Helper function to create a menu with all features.
pub fn create_full_menu(
    mode_signal: Arc<AtomicU8>,
    action_sender: mpsc::Sender<MenuAction>,
) -> impl FnMut(&Context) + Send + 'static {
    MenuBuilder::new()
        .with_mode_signal(mode_signal)
        .with_action_sender(action_sender)
        .build()
}

/// Read the current mode from the signal.
fn read_mode(signal: &Arc<AtomicU8>) -> LayerMode {
    match signal.load(Ordering::Relaxed) {
        0 => LayerMode::Points,
        1 => LayerMode::Gaussians,
        2 => LayerMode::Triangles,
        _ => LayerMode::Points,
    }
}

/// Write a mode to the signal.
fn write_mode(signal: &Arc<AtomicU8>, mode: LayerMode) {
    let value = match mode {
        LayerMode::Points => 0,
        LayerMode::Gaussians => 1,
        LayerMode::Triangles => 2,
    };
    signal.store(value, Ordering::Relaxed);
}

/// List available camera devices.
fn list_camera_devices() -> Result<Vec<String>, String> {
    // Use WebcamCapture to list devices
    // This requires the webcam feature to be enabled in triad-capture
    use triad_capture::WebcamCapture;
    WebcamCapture::list_devices().map_err(|e| e.to_string())
}
