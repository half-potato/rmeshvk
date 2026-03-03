pub mod shared;

// WGSL shader sources, embedded as string constants.
pub const FORWARD_COMPUTE_WGSL: &str = include_str!("wgsl/forward_compute.wgsl");
pub const FORWARD_VERTEX_WGSL: &str = include_str!("wgsl/forward_vertex.wgsl");
pub const FORWARD_FRAGMENT_WGSL: &str = include_str!("wgsl/forward_fragment.wgsl");
pub const RADIX_SORT_WGSL: &str = include_str!("wgsl/radix_sort.wgsl");
pub const LOSS_COMPUTE_WGSL: &str = include_str!("wgsl/loss_compute.wgsl");
pub const BACKWARD_COMPUTE_WGSL: &str = include_str!("wgsl/backward_compute.wgsl");
pub const ADAM_COMPUTE_WGSL: &str = include_str!("wgsl/adam_compute.wgsl");
pub const TEX_TO_BUFFER_WGSL: &str = include_str!("wgsl/tex_to_buffer.wgsl");
