//! naga_oil Composer helper for shared WGSL modules.
//!
//! Provides shared WGSL modules (math, SH constants, intersection helpers)
//! and a helper to create composed shader modules via naga_oil's Composer.
//!
//! ## naga_oil 0.21 caveats
//!
//! - **Selective imports only**: Use `#import rmesh::math::{phi, MAX_VAL}`,
//!   NOT `#import rmesh::math` (plain imports silently fail to inject defs).
//! - **No struct imports**: Modules containing struct definitions (like
//!   `rmesh::common`) cause `InvalidIdentifier` errors on struct member names.
//!   Struct definitions must be inlined in each shader.

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage, ShaderType,
};

const MATH_WGSL: &str = include_str!("wgsl/math.wgsl");
const SH_WGSL: &str = include_str!("wgsl/sh.wgsl");
const INTERSECT_WGSL: &str = include_str!("wgsl/intersect.wgsl");

/// Create a Composer pre-loaded with all shared rmesh WGSL modules.
///
/// Modules available for selective import:
/// - `rmesh::math` — MAX_VAL, safe_clip_v3f, safe_exp_f32, phi, dphi_dx, softplus, dsoftplus, project_to_ndc
/// - `rmesh::sh` — SH basis constants (C0, C1, C2_*, C3_*)
/// - `rmesh::intersect` — FACES constant
///
/// Use selective import syntax: `#import rmesh::math::{phi, MAX_VAL}`
pub fn create_composer() -> Result<Composer, String> {
    let mut composer = Composer::default();

    let modules = [
        ("rmesh::math", MATH_WGSL),
        ("rmesh::sh", SH_WGSL),
        ("rmesh::intersect", INTERSECT_WGSL),
    ];

    for (name, source) in modules {
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source,
                file_path: name,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .map_err(|e| format!("Failed to add module {name}: {e:?}"))?;
    }

    Ok(composer)
}

/// Compose a WGSL shader source (which may `#import` shared modules) into a
/// `wgpu::ShaderModule`.
///
/// The source should use selective imports:
/// `#import rmesh::math::{phi, MAX_VAL}`
pub fn create_shader_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> Result<wgpu::ShaderModule, String> {
    let mut composer = create_composer()?;

    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path: label,
            shader_type: ShaderType::Wgsl,
            ..Default::default()
        })
        .map_err(|e| format!("Failed to compose shader {label}: {e:?}"))?;

    // Convert the composed naga module back to WGSL text, since wgpu 28
    // no longer accepts ShaderSource::Naga directly.
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("Validation error in {label}: {e:?}"))?;

    let wgsl_text = naga::back::wgsl::write_string(
        &module,
        &info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .map_err(|e| format!("WGSL write error in {label}: {e:?}"))?;

    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl_text.into()),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_selective_math() {
        let mut composer = create_composer().expect("create_composer failed");

        let source = r#"
#import rmesh::math::{phi, safe_exp_f32, MAX_VAL, safe_clip_v3f}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main() {
    out[0] = phi(1.0);
    out[1] = safe_exp_f32(1.0);
    out[2] = MAX_VAL;
    let v = safe_clip_v3f(vec3<f32>(1.0), 0.0, 1.0);
    out[3] = v.x;
}
"#;

        let module = composer.make_naga_module(NagaModuleDescriptor {
            source,
            file_path: "test_math.wgsl",
            shader_type: ShaderType::Wgsl,
            ..Default::default()
        });
        assert!(module.is_ok(), "selective math import failed: {:?}", module.err());
    }

    #[test]
    fn test_compose_selective_intersect() {
        let mut composer = create_composer().expect("create_composer failed");

        let source = r#"
#import rmesh::intersect::FACES

@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let f = FACES[0];
    out[0] = f.x;
}
"#;

        let module = composer.make_naga_module(NagaModuleDescriptor {
            source,
            file_path: "test_intersect.wgsl",
            shader_type: ShaderType::Wgsl,
            ..Default::default()
        });
        assert!(module.is_ok(), "selective intersect import failed: {:?}", module.err());
    }
}
