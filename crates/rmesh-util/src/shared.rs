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
    /// Camera-to-world rotation matrix columns (xyz, w=0 padding)
    pub c2w_col0: [f32; 4],
    pub c2w_col1: [f32; 4],
    pub c2w_col2: [f32; 4],
    /// Camera intrinsics [fx, fy, cx, cy]
    pub intrinsics: [f32; 4],
    /// Camera position (xyz) + padding (w)
    pub cam_pos_pad: [f32; 4],
    /// Screen dimensions
    pub screen_width: f32,
    pub screen_height: f32,
    /// Total number of tetrahedra
    pub tet_count: u32,
    /// Current training step
    pub step: u32,
    /// Tile size for tiled pipeline (must match shaders).
    pub tile_size_u: u32,
    /// Ray mode: 0 = derive from c2w + intrinsics, 1 = read from ray buffers.
    pub ray_mode: u32,
    /// Minimum ray-origin offset along view direction (matches Slang camera.min_t).
    pub min_t: f32,
    /// SH degree for inline SH evaluation (0 = use base_colors path, 1-3 = eval SH).
    pub sh_degree: u32,
    /// Near clip plane distance (positive, view-space Z).
    pub near_plane: f32,
    /// Far clip plane distance (positive, view-space Z).
    pub far_plane: f32,
    /// Padding to maintain 192-byte struct size.
    pub _pad1: [u32; 2],
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

    pub fn c2w_matrix(&self) -> glam::Mat3 {
        glam::Mat3::from_cols(
            glam::Vec3::new(self.c2w_col0[0], self.c2w_col0[1], self.c2w_col0[2]),
            glam::Vec3::new(self.c2w_col1[0], self.c2w_col1[1], self.c2w_col1[2]),
            glam::Vec3::new(self.c2w_col2[0], self.c2w_col2[1], self.c2w_col2[2]),
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
