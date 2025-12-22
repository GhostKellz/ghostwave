//! GUI editor for GhostWave VST plugin - NVIDIA Broadcast Style
//!
//! Clean, modern interface inspired by NVIDIA Broadcast and WaveLink XLR.
//! Features:
//! - Large toggle switches for main features
//! - Real-time level meters
//! - RTX acceleration status
//! - Minimal, professional dark theme

use crate::params::GhostWaveParams;
use nih_plug::prelude::*;
use nih_plug_egui::egui::{self, Color32, CornerRadius, Margin, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use nih_plug_egui::{create_egui_editor, EguiState};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// Colors - NVIDIA Broadcast inspired
const BG_DARK: Color32 = Color32::from_rgb(18, 18, 22);
const BG_PANEL: Color32 = Color32::from_rgb(28, 28, 34);
const BG_CARD: Color32 = Color32::from_rgb(38, 38, 46);
const ACCENT_GREEN: Color32 = Color32::from_rgb(118, 185, 0); // NVIDIA Green
const ACCENT_BLUE: Color32 = Color32::from_rgb(0, 168, 232);
const TEXT_PRIMARY: Color32 = Color32::from_rgb(240, 240, 245);
const TEXT_SECONDARY: Color32 = Color32::from_rgb(160, 160, 170);
const TEXT_MUTED: Color32 = Color32::from_rgb(100, 100, 110);
const METER_GREEN: Color32 = Color32::from_rgb(76, 175, 80);
const METER_YELLOW: Color32 = Color32::from_rgb(255, 193, 7);
const METER_RED: Color32 = Color32::from_rgb(244, 67, 54);

/// Shared state for metering
pub struct EditorState {
    input_level: AtomicU32,
    output_level: AtomicU32,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            input_level: AtomicU32::new(0),
            output_level: AtomicU32::new(0),
        }
    }
}

/// Create the plugin editor
pub fn create(
    params: Arc<GhostWaveParams>,
    editor_state: Arc<EguiState>,
) -> Option<Box<dyn Editor>> {
    let state = Arc::new(EditorState::default());

    create_egui_editor(
        editor_state,
        state,
        |_, _| {},
        move |egui_ctx, setter, state| {
            draw_ui(egui_ctx, setter, &params, state);
        },
    )
}

/// Main UI drawing function
fn draw_ui(
    ctx: &egui::Context,
    setter: &ParamSetter,
    params: &GhostWaveParams,
    state: &EditorState,
) {
    // Apply dark theme
    let mut style = (*ctx.style()).clone();
    style.visuals.dark_mode = true;
    style.visuals.panel_fill = BG_DARK;
    style.visuals.window_fill = BG_PANEL;
    style.visuals.widgets.noninteractive.bg_fill = BG_CARD;
    style.visuals.widgets.inactive.bg_fill = BG_CARD;
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(50, 50, 60);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(60, 60, 75);
    style.spacing.item_spacing = Vec2::new(8.0, 8.0);
    ctx.set_style(style);

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(BG_DARK).inner_margin(16))
        .show(ctx, |ui| {
            // Header with logo and RTX status
            draw_header(ui);

            ui.add_space(16.0);

            // Main content: two columns
            ui.horizontal(|ui| {
                // Left column: Main effects
                ui.vertical(|ui| {
                    ui.set_min_width(340.0);

                    // Noise Suppression - Main feature card
                    draw_feature_card(
                        ui,
                        setter,
                        "NOISE REMOVAL",
                        "AI-powered noise suppression",
                        ACCENT_GREEN,
                        &params.noise_enabled,
                        |ui, setter| {
                            draw_slider(
                                ui,
                                setter,
                                "Strength",
                                &params.noise_strength,
                                |v| format!("{:.0}%", v * 100.0),
                            );
                            draw_slider(
                                ui,
                                setter,
                                "Gate",
                                &params.noise_gate_threshold,
                                |v| format!("{:.0} dB", v),
                            );
                        },
                    );

                    ui.add_space(12.0);

                    // Echo Removal
                    draw_feature_card(
                        ui,
                        setter,
                        "ROOM ECHO REMOVAL",
                        "Reduce room reverb and echo",
                        ACCENT_BLUE,
                        &params.echo_enabled,
                        |ui, setter| {
                            draw_slider(
                                ui,
                                setter,
                                "Strength",
                                &params.echo_strength,
                                |v| format!("{:.0}%", v * 100.0),
                            );
                        },
                    );

                    ui.add_space(12.0);

                    // De-Esser
                    draw_feature_card(
                        ui,
                        setter,
                        "DE-ESSER",
                        "Reduce harsh sibilance",
                        Color32::from_rgb(156, 39, 176),
                        &params.deesser_enabled,
                        |ui, setter| {
                            draw_slider(
                                ui,
                                setter,
                                "Threshold",
                                &params.deesser_threshold,
                                |v| format!("{:.0} dB", v),
                            );
                            draw_slider(
                                ui,
                                setter,
                                "Frequency",
                                &params.deesser_frequency,
                                |v| format!("{:.0} Hz", v),
                            );
                        },
                    );
                });

                ui.add_space(16.0);

                // Right column: Dynamics + Output
                ui.vertical(|ui| {
                    ui.set_min_width(340.0);

                    // Compressor
                    draw_feature_card(
                        ui,
                        setter,
                        "COMPRESSOR",
                        "Dynamic range control",
                        Color32::from_rgb(255, 152, 0),
                        &params.comp_enabled,
                        |ui, setter| {
                            draw_slider(
                                ui,
                                setter,
                                "Threshold",
                                &params.comp_threshold,
                                |v| format!("{:.0} dB", v),
                            );
                            draw_slider(
                                ui,
                                setter,
                                "Ratio",
                                &params.comp_ratio,
                                |v| format!("{:.1}:1", v),
                            );
                            draw_slider(
                                ui,
                                setter,
                                "Attack",
                                &params.comp_attack,
                                |v| format!("{:.1} ms", v),
                            );
                            draw_slider(
                                ui,
                                setter,
                                "Release",
                                &params.comp_release,
                                |v| format!("{:.0} ms", v),
                            );
                        },
                    );

                    ui.add_space(12.0);

                    // Limiter
                    draw_feature_card(
                        ui,
                        setter,
                        "LIMITER",
                        "Prevent clipping",
                        METER_RED,
                        &params.limiter_enabled,
                        |ui, setter| {
                            draw_slider(
                                ui,
                                setter,
                                "Ceiling",
                                &params.limiter_ceiling,
                                |v| format!("{:.1} dB", v),
                            );
                        },
                    );

                    ui.add_space(12.0);

                    // Output section with meters
                    draw_output_section(ui, setter, params, state);
                });
            });

            // Footer
            ui.add_space(12.0);
            draw_footer(ui, params);
        });

    // Request repaint for animations (meters)
    ctx.request_repaint();
}

/// Draw the header with logo and RTX status
fn draw_header(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        // Logo
        ui.vertical(|ui| {
            ui.label(
                RichText::new("GHOSTWAVE")
                    .size(24.0)
                    .color(TEXT_PRIMARY)
                    .strong(),
            );
            ui.label(
                RichText::new("AI Audio Processing")
                    .size(11.0)
                    .color(TEXT_MUTED),
            );
        });

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // RTX Status indicator
            draw_rtx_status(ui);
        });
    });
}

/// Draw RTX acceleration status badge
fn draw_rtx_status(ui: &mut egui::Ui) {
    let rtx_available = ghostwave_core::cuda_runtime::CudaRuntime::is_available();

    let (status_text, status_color, bg_color) = if rtx_available {
        ("RTX ON", ACCENT_GREEN, Color32::from_rgb(30, 50, 30))
    } else {
        ("CPU MODE", TEXT_MUTED, BG_CARD)
    };

    let frame = egui::Frame::NONE
        .fill(bg_color)
        .stroke(Stroke::new(1.0, status_color.gamma_multiply(0.6)))
        .corner_radius(CornerRadius::same(4))
        .inner_margin(Margin::symmetric(12, 6));

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            // GPU icon (simple rectangle representation)
            let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 12.0), Sense::hover());
            ui.painter().rect_filled(rect, CornerRadius::same(2), status_color);
            ui.painter().rect_stroke(
                rect.shrink(2.0),
                CornerRadius::same(1),
                Stroke::new(1.0, BG_DARK),
                egui::StrokeKind::Outside,
            );

            ui.label(RichText::new(status_text).size(11.0).color(status_color).strong());
        });
    });
}

/// Draw a feature card with toggle and controls
fn draw_feature_card(
    ui: &mut egui::Ui,
    setter: &ParamSetter,
    title: &str,
    subtitle: &str,
    accent: Color32,
    enabled_param: &BoolParam,
    add_controls: impl FnOnce(&mut egui::Ui, &ParamSetter),
) {
    let enabled = enabled_param.value();
    let card_alpha = if enabled { 1.0 } else { 0.6 };

    let frame = egui::Frame::NONE
        .fill(BG_PANEL)
        .stroke(Stroke::new(
            1.0,
            if enabled {
                accent.gamma_multiply(0.4)
            } else {
                Color32::from_rgb(50, 50, 55)
            },
        ))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(16);

    frame.show(ui, |ui| {
        // Header with title and toggle
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(title)
                        .size(13.0)
                        .color(if enabled { accent } else { TEXT_MUTED })
                        .strong(),
                );
                ui.label(
                    RichText::new(subtitle)
                        .size(10.0)
                        .color(TEXT_MUTED.gamma_multiply(card_alpha)),
                );
            });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_toggle(ui, setter, enabled_param, accent);
            });
        });

        // Controls (only interactive when enabled)
        if enabled {
            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);
            add_controls(ui, setter);
        }
    });
}

/// Draw a custom toggle switch
fn draw_toggle(ui: &mut egui::Ui, setter: &ParamSetter, param: &BoolParam, accent: Color32) {
    let enabled = param.value();
    let desired_size = Vec2::new(48.0, 24.0);

    let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click());

    if response.clicked() {
        setter.set_parameter(param, !enabled);
    }

    // Animate toggle position
    let anim_t = ui.ctx().animate_bool(response.id, enabled);

    // Background pill
    let bg_color = if enabled {
        accent
    } else {
        Color32::from_rgb(60, 60, 70)
    };
    ui.painter()
        .rect_filled(rect, CornerRadius::same(12), bg_color);

    // Toggle circle
    let circle_radius = 10.0;
    let circle_x = egui::lerp(
        rect.left() + circle_radius + 2.0..=rect.right() - circle_radius - 2.0,
        anim_t,
    );
    let circle_center = Pos2::new(circle_x, rect.center().y);

    ui.painter().circle_filled(
        circle_center,
        circle_radius,
        if enabled { TEXT_PRIMARY } else { TEXT_SECONDARY },
    );

    // Hover effect
    if response.hovered() {
        ui.painter().rect_stroke(
            rect,
            CornerRadius::same(12),
            Stroke::new(1.0, TEXT_PRIMARY.gamma_multiply(0.3)),
            egui::StrokeKind::Outside,
        );
    }
}

/// Draw a slider control
fn draw_slider(
    ui: &mut egui::Ui,
    setter: &ParamSetter,
    label: &str,
    param: &FloatParam,
    format_value: impl Fn(f32) -> String,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.0).color(TEXT_SECONDARY));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Value display
            let value = param.value();
            ui.label(
                RichText::new(format_value(value))
                    .size(11.0)
                    .color(TEXT_PRIMARY)
                    .strong(),
            );
        });
    });

    ui.add_space(4.0);

    // Custom slider
    let _range = param.default_plain_value()..(param.default_plain_value() * 2.0); // Placeholder
    let slider_height = 6.0;
    let desired_size = Vec2::new(ui.available_width(), slider_height + 16.0);

    let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click_and_drag());

    let slider_rect = Rect::from_center_size(rect.center(), Vec2::new(rect.width(), slider_height));

    // Get normalized value
    let normalized = param.modulated_normalized_value();

    // Handle interaction
    if response.dragged() || response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let new_normalized =
                ((pos.x - slider_rect.left()) / slider_rect.width()).clamp(0.0, 1.0);
            setter.set_parameter_normalized(param, new_normalized);
        }
    }

    // Draw slider background
    ui.painter()
        .rect_filled(slider_rect, CornerRadius::same(3), BG_DARK);

    // Draw filled portion
    let filled_width = slider_rect.width() * normalized;
    let filled_rect = Rect::from_min_size(slider_rect.min, Vec2::new(filled_width, slider_height));
    ui.painter()
        .rect_filled(filled_rect, CornerRadius::same(3), ACCENT_GREEN);

    // Draw handle
    let handle_x = slider_rect.left() + filled_width;
    let handle_center = Pos2::new(handle_x, slider_rect.center().y);
    let handle_radius = if response.hovered() || response.dragged() {
        8.0
    } else {
        6.0
    };

    ui.painter()
        .circle_filled(handle_center, handle_radius, TEXT_PRIMARY);

    if response.hovered() || response.dragged() {
        ui.painter().circle_stroke(
            handle_center,
            handle_radius + 2.0,
            Stroke::new(2.0, ACCENT_GREEN.gamma_multiply(0.5)),
        );
    }

    ui.add_space(4.0);
}

/// Draw output section with level meters
fn draw_output_section(
    ui: &mut egui::Ui,
    setter: &ParamSetter,
    params: &GhostWaveParams,
    state: &EditorState,
) {
    let frame = egui::Frame::NONE
        .fill(BG_PANEL)
        .stroke(Stroke::new(1.0, Color32::from_rgb(50, 50, 55)))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(16);

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("OUTPUT")
                .size(13.0)
                .color(TEXT_PRIMARY)
                .strong(),
        );

        ui.add_space(12.0);

        // Level meters
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("IN").size(9.0).color(TEXT_MUTED));
                draw_level_meter(ui, state.input_level.load(Ordering::Relaxed) as f32 / 100.0);
            });

            ui.add_space(8.0);

            ui.vertical(|ui| {
                ui.label(RichText::new("OUT").size(9.0).color(TEXT_MUTED));
                draw_level_meter(ui, state.output_level.load(Ordering::Relaxed) as f32 / 100.0);
            });

            ui.add_space(16.0);

            ui.vertical(|ui| {
                draw_slider(
                    ui,
                    setter,
                    "Output Gain",
                    &params.output_gain,
                    |v| format!("{:+.1} dB", v),
                );

                ui.add_space(8.0);

                draw_slider(
                    ui,
                    setter,
                    "Dry/Wet Mix",
                    &params.dry_wet,
                    |v| format!("{:.0}%", v * 100.0),
                );
            });
        });
    });
}

/// Draw a vertical level meter
fn draw_level_meter(ui: &mut egui::Ui, level: f32) {
    let desired_size = Vec2::new(20.0, 80.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, Sense::hover());

    // Background
    ui.painter()
        .rect_filled(rect, CornerRadius::same(2), BG_DARK);

    // Segments
    let num_segments = 16;
    let segment_height = rect.height() / num_segments as f32;
    let gap = 1.0;

    for i in 0..num_segments {
        let segment_level = (num_segments - i) as f32 / num_segments as f32;
        let is_lit = level >= segment_level - (1.0 / num_segments as f32);

        let color = if i < 2 {
            if is_lit { METER_RED } else { METER_RED.gamma_multiply(0.2) }
        } else if i < 5 {
            if is_lit { METER_YELLOW } else { METER_YELLOW.gamma_multiply(0.2) }
        } else if is_lit {
            METER_GREEN
        } else {
            METER_GREEN.gamma_multiply(0.2)
        };

        let segment_rect = Rect::from_min_size(
            Pos2::new(rect.left() + 2.0, rect.top() + i as f32 * segment_height + gap),
            Vec2::new(rect.width() - 4.0, segment_height - gap * 2.0),
        );

        ui.painter()
            .rect_filled(segment_rect, CornerRadius::same(1), color);
    }
}

/// Draw the footer
fn draw_footer(ui: &mut egui::Ui, params: &GhostWaveParams) {
    ui.separator();
    ui.add_space(4.0);

    ui.horizontal(|ui| {
        ui.label(
            RichText::new("Ghost Ecosystem")
                .size(10.0)
                .color(TEXT_MUTED),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                    .size(10.0)
                    .color(TEXT_MUTED),
            );

            // EQ indicator
            if params.eq_enabled.value() {
                ui.label(
                    RichText::new("EQ")
                        .size(9.0)
                        .color(ACCENT_BLUE)
                        .strong(),
                );
            }
        });
    });
}
