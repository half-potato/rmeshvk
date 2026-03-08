/// Types shared between CPU and GPU code.
/// These match the WGSL struct layouts byte-for-byte (std430/storage buffer layout).
///
/// Matches the FrozenTetModel layout from delaunay_splatting:
///   - Per-tet: density (1), rgb (3), gradient (3×3), sh ((deg+1)^2 × 3)
///   - No iNGP network — direct parameter storage

use bytemuck::{Pod, Zeroable};

/// Per-frame uniforms.
/// Bound as a **storage buffer** (not uniform) to avoid std140 alignment hassles.
/// std430 layout matches Rust's repr(C) directly.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    /// Column-major 4×4 view-projection matrix
    pub vp_col0: [f32; 4],
    pub vp_col1: [f32; 4],
    pub vp_col2: [f32; 4],
    pub vp_col3: [f32; 4],
    /// Column-major 4×4 inverse view-projection matrix (for backward pass ray computation)
    pub inv_vp_col0: [f32; 4],
    pub inv_vp_col1: [f32; 4],
    pub inv_vp_col2: [f32; 4],
    pub inv_vp_col3: [f32; 4],
    /// Camera position (xyz) + padding (w)
    pub cam_pos_pad: [f32; 4],
    /// Screen dimensions
    pub screen_width: f32,
    pub screen_height: f32,
    /// Total number of tetrahedra
    pub tet_count: u32,
    /// SH degree (0-3)
    pub sh_degree: u32,
    /// Current training step
    pub step: u32,
    /// Padding: 3 u32s to reach vec3<u32> alignment (offset 176), then 3 u32s
    /// for the vec3 itself, then 1 u32 for struct-end alignment to 192 bytes.
    pub _pad1: [u32; 7],
}

impl Uniforms {
    pub fn vp_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_cols(
            glam::Vec4::from(self.vp_col0),
            glam::Vec4::from(self.vp_col1),
            glam::Vec4::from(self.vp_col2),
            glam::Vec4::from(self.vp_col3),
        )
    }

    pub fn inv_vp_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_cols(
            glam::Vec4::from(self.inv_vp_col0),
            glam::Vec4::from(self.inv_vp_col1),
            glam::Vec4::from(self.inv_vp_col2),
            glam::Vec4::from(self.inv_vp_col3),
        )
    }
}

/// DrawIndirectCommand layout (matches wgpu::DrawIndirectArgs).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Loss computation uniforms.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LossUniforms {
    pub width: u32,
    pub height: u32,
    /// 0 = L1, 1 = L2, 2 = SSIM
    pub loss_type: u32,
    /// Weight for SSIM component when using mixed loss
    pub lambda_ssim: f32,
}

/// Adam optimizer push constants / uniforms.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AdamUniforms {
    pub param_count: u32,
    pub step: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub _pad: [u32; 2],
}

/// Bitonic sort uniforms.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SortUniforms {
    pub count: u32,
    /// Bitonic sort stage (outer loop)
    pub stage: u32,
    /// Bitonic sort step within stage (inner loop)
    pub step_size: u32,
    pub _pad: u32,
}

// Constant face winding for all tets (4 faces × 3 vertex indices)
// 12 vertices total when unrolled for triangle-list topology
pub const TET_FACE_INDICES: [u32; 12] = [
    0, 2, 1, // face 0
    1, 2, 3, // face 1
    0, 3, 2, // face 2
    3, 0, 1, // face 3
];

// --- SH basis constants ---
pub const C0: f32 = 0.28209479177387814;
pub const C1: f32 = 0.4886025119029199;

pub const C2_0: f32 = 1.0925484305920792;
pub const C2_1: f32 = -1.0925484305920792;
pub const C2_2: f32 = 0.31539156525252005;
pub const C2_3: f32 = -1.0925484305920792;
pub const C2_4: f32 = 0.5462742152960396;

pub const C3_0: f32 = -0.5900435899266435;
pub const C3_1: f32 = 2.890611442640554;
pub const C3_2: f32 = -0.4570457994644658;
pub const C3_3: f32 = 0.3731763325901154;
pub const C3_4: f32 = -0.4570457994644658;
pub const C3_5: f32 = 1.445305721320277;
pub const C3_6: f32 = -0.5900435899266435;

/// Tile-based backward pass uniforms.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TileUniforms {
    pub screen_width: u32,
    pub screen_height: u32,
    pub tile_size: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub num_tiles: u32,
    pub visible_tet_count: u32,
    pub _pad: [u32; 5],
}

/// BVH node for boundary face acceleration structure.
/// Matches the WGSL `BVHNode` struct layout (std430).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BVHNode {
    pub aabb_min: [f32; 3],
    /// >= 0: left child index, < 0: -(leaf_start + 1)
    pub left_or_face: i32,
    pub aabb_max: [f32; 3],
    /// Internal: right child index. Leaf: face count.
    pub right_or_count: i32,
}

// Safe math constants (from safe_math.slang)
pub const SAFE_MIN: f32 = -1e20;
pub const SAFE_MAX: f32 = 1e20;
pub const TINY_VAL: f32 = 1e-20;
/// Early termination threshold: exp(-5.54) ≈ 1/255
pub const LOG_TRANSMITTANCE_THRESHOLD: f32 = -5.54;
