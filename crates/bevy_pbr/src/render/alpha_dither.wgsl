#define_import_path bevy_pbr::alpha_dither

#import bevy_pbr::{mesh_view_bindings::view, utils::interleaved_gradient_noise}

// The cutoff of the alpha test that stands in for alpha to coverage without MSAA. Each pixel
// tests against a value near `cutoff` that varies with its position and over a cycle of eight
// frames, so a temporal anti-aliasing pass averages the edges of cut-out surfaces, such as
// leaves. The prepass and the main pass both bind the view at binding 0 and call this with
// the same pixel, so they keep and drop the same pixels.
fn alpha_to_coverage_cutoff(frag_coord: vec2<f32>, cutoff: f32) -> f32 {
    let noise = interleaved_gradient_noise(frag_coord, view.frame_count % 8u);
    return clamp(cutoff + (noise - 0.5) * 0.5, 0.004, 1.0);
}
