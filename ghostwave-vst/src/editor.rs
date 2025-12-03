//! GUI editor for GhostWave VST plugin
//!
//! Built with egui via nih_plug_egui.
//! Professional dark theme with clear section organization.
//!
//! Note: This is a read-only display for now. Parameter changes
//! are handled through host automation. Full GUI control
//! requires additional nih-plug setter integration.

use crate::params::GhostWaveParams;
use nih_plug::prelude::*;
use nih_plug_egui::egui::{self, Color32, CornerRadius, RichText, Stroke};
use nih_plug_egui::{create_egui_editor, EguiState};
use std::sync::Arc;

/// Create the plugin editor
pub fn create(
    params: Arc<GhostWaveParams>,
    editor_state: Arc<EguiState>,
) -> Option<Box<dyn Editor>> {
    create_egui_editor(
        editor_state,
        (),
        |_, _| {},
        move |egui_ctx, _setter, _state| {
            draw_ui(egui_ctx, &params);
        },
    )
}

/// Main UI drawing function
fn draw_ui(ctx: &egui::Context, params: &GhostWaveParams) {
    // Apply dark theme
    let mut style = (*ctx.style()).clone();
    style.visuals.dark_mode = true;
    style.visuals.panel_fill = Color32::from_rgb(25, 25, 30);
    style.visuals.window_fill = Color32::from_rgb(30, 30, 35);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(40, 40, 48);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(50, 50, 60);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(60, 60, 75);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(80, 80, 100);
    ctx.set_style(style);

    egui::CentralPanel::default().show(ctx, |ui| {
        // Header
        ui.horizontal(|ui| {
            ui.heading(
                RichText::new("GhostWave")
                    .size(28.0)
                    .color(Color32::from_rgb(100, 200, 255)),
            );
            ui.add_space(20.0);
            ui.label(
                RichText::new("NVIDIA Broadcast for Linux")
                    .size(12.0)
                    .color(Color32::GRAY),
            );
        });
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);

        // Main content in columns
        ui.horizontal(|ui| {
            // Left column: Noise + Echo + De-Esser
            ui.vertical(|ui| {
                ui.set_min_width(250.0);

                // Noise Suppression Section
                draw_section(
                    ui,
                    "Noise Suppression",
                    Color32::from_rgb(100, 200, 100),
                    |ui| {
                        draw_bool_display(ui, "Enabled", params.noise_enabled.value());
                        draw_value_display(
                            ui,
                            "Strength",
                            params.noise_strength.value() * 100.0,
                            "%",
                        );
                        draw_value_display(ui, "Gate", params.noise_gate_threshold.value(), "dB");
                    },
                );

                ui.add_space(15.0);

                // Echo Removal Section
                draw_section(ui, "Echo Removal", Color32::from_rgb(200, 150, 100), |ui| {
                    draw_bool_display(ui, "Enabled", params.echo_enabled.value());
                    draw_value_display(
                        ui,
                        "Strength",
                        params.echo_strength.value() * 100.0,
                        "%",
                    );
                });

                ui.add_space(15.0);

                // De-Esser Section
                draw_section(ui, "De-Esser", Color32::from_rgb(200, 100, 150), |ui| {
                    draw_bool_display(ui, "Enabled", params.deesser_enabled.value());
                    draw_value_display(ui, "Threshold", params.deesser_threshold.value(), "dB");
                    draw_value_display(ui, "Frequency", params.deesser_frequency.value(), "Hz");
                    draw_value_display(ui, "Ratio", params.deesser_ratio.value(), ":1");
                });
            });

            ui.add_space(20.0);

            // Middle column: EQ
            ui.vertical(|ui| {
                ui.set_min_width(250.0);

                draw_section(ui, "Parametric EQ", Color32::from_rgb(100, 150, 200), |ui| {
                    draw_bool_display(ui, "Enabled", params.eq_enabled.value());
                    ui.add_space(5.0);

                    // EQ Bands
                    draw_eq_band_display(
                        ui,
                        "Low Shelf",
                        params.eq_band1_enabled.value(),
                        params.eq_band1_freq.value(),
                        params.eq_band1_gain.value(),
                        None,
                    );
                    draw_eq_band_display(
                        ui,
                        "Low-Mid",
                        params.eq_band2_enabled.value(),
                        params.eq_band2_freq.value(),
                        params.eq_band2_gain.value(),
                        Some(params.eq_band2_q.value()),
                    );
                    draw_eq_band_display(
                        ui,
                        "Mid",
                        params.eq_band3_enabled.value(),
                        params.eq_band3_freq.value(),
                        params.eq_band3_gain.value(),
                        Some(params.eq_band3_q.value()),
                    );
                    draw_eq_band_display(
                        ui,
                        "High-Mid",
                        params.eq_band4_enabled.value(),
                        params.eq_band4_freq.value(),
                        params.eq_band4_gain.value(),
                        Some(params.eq_band4_q.value()),
                    );
                    draw_eq_band_display(
                        ui,
                        "High Shelf",
                        params.eq_band5_enabled.value(),
                        params.eq_band5_freq.value(),
                        params.eq_band5_gain.value(),
                        None,
                    );
                });
            });

            ui.add_space(20.0);

            // Right column: Compressor + Limiter + Output
            ui.vertical(|ui| {
                ui.set_min_width(250.0);

                // Compressor Section
                draw_section(ui, "Compressor", Color32::from_rgb(200, 200, 100), |ui| {
                    draw_bool_display(ui, "Enabled", params.comp_enabled.value());
                    draw_value_display(ui, "Threshold", params.comp_threshold.value(), "dB");
                    draw_value_display(ui, "Ratio", params.comp_ratio.value(), ":1");
                    draw_value_display(ui, "Attack", params.comp_attack.value(), "ms");
                    draw_value_display(ui, "Release", params.comp_release.value(), "ms");
                    draw_value_display(ui, "Knee", params.comp_knee.value(), "dB");
                    draw_value_display(ui, "Makeup", params.comp_makeup.value(), "dB");
                });

                ui.add_space(15.0);

                // Limiter Section
                draw_section(ui, "Limiter", Color32::from_rgb(255, 100, 100), |ui| {
                    draw_bool_display(ui, "Enabled", params.limiter_enabled.value());
                    draw_value_display(ui, "Ceiling", params.limiter_ceiling.value(), "dB");
                });

                ui.add_space(15.0);

                // Output Section
                draw_section(ui, "Output", Color32::from_rgb(150, 150, 200), |ui| {
                    draw_value_display(ui, "Gain", params.output_gain.value(), "dB");
                    draw_value_display(ui, "Dry/Wet", params.dry_wet.value() * 100.0, "%");
                });
            });
        });

        // Footer
        ui.add_space(10.0);
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Ghost Ecosystem").size(10.0).color(Color32::GRAY));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                        .size(10.0)
                        .color(Color32::GRAY),
                );
            });
        });
    });
}

/// Draw a collapsible section with header
fn draw_section<R>(
    ui: &mut egui::Ui,
    title: &str,
    accent_color: Color32,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) {
    let frame = egui::Frame::default()
        .fill(Color32::from_rgb(35, 35, 42))
        .stroke(Stroke::new(1.0, accent_color.gamma_multiply(0.5)))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(10.0);

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(title).size(14.0).color(accent_color).strong());
        });
        ui.add_space(8.0);
        add_contents(ui);
    });
}

/// Draw a boolean parameter display
fn draw_bool_display(ui: &mut egui::Ui, label: &str, value: bool) {
    ui.horizontal(|ui| {
        let status = if value { "ON" } else { "OFF" };
        let color = if value {
            Color32::from_rgb(100, 200, 100)
        } else {
            Color32::GRAY
        };
        ui.label(RichText::new(label).size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(status).size(11.0).color(color));
        });
    });
}

/// Draw a value display
fn draw_value_display(ui: &mut egui::Ui, label: &str, value: f32, unit: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(format!("{:.1} {}", value, unit))
                    .size(11.0)
                    .color(Color32::from_rgb(180, 180, 200)),
            );
        });
    });
}

/// Draw an EQ band display with compact layout
fn draw_eq_band_display(
    ui: &mut egui::Ui,
    name: &str,
    enabled: bool,
    freq: f32,
    gain: f32,
    q: Option<f32>,
) {
    let status_color = if enabled {
        Color32::from_rgb(100, 200, 100)
    } else {
        Color32::GRAY
    };

    ui.horizontal(|ui| {
        ui.label(RichText::new(name).size(10.0).color(status_color));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let q_str = q.map(|v| format!(" Q:{:.1}", v)).unwrap_or_default();
            ui.label(
                RichText::new(format!("{:.0}Hz {:+.1}dB{}", freq, gain, q_str))
                    .size(10.0)
                    .color(Color32::from_rgb(150, 150, 170)),
            );
        });
    });
}
