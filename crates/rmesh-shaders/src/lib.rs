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
pub const TILE_FILL_WGSL: &str = include_str!("wgsl/tile_fill_compute.wgsl");
pub const TILE_GEN_WGSL: &str = include_str!("wgsl/tile_gen_compute.wgsl");
pub const TILE_GEN_HULL_WGSL: &str = include_str!("wgsl/tile_gen_hull_compute.wgsl");
pub const TILE_RANGES_WGSL: &str = include_str!("wgsl/tile_ranges_compute.wgsl");
pub const BACKWARD_TILED_WGSL: &str = include_str!("wgsl/backward_tiled_compute.wgsl");
pub const FORWARD_TILED_WGSL: &str = include_str!("wgsl/forward_tiled_compute.wgsl");
pub const PREPARE_DISPATCH_WGSL: &str = include_str!("wgsl/prepare_dispatch.wgsl");
pub const PREFIX_SCAN_WGSL: &str = include_str!("wgsl/prefix_scan.wgsl");
pub const PREFIX_SCAN_ADD_WGSL: &str = include_str!("wgsl/prefix_scan_add.wgsl");
pub const TILE_GEN_SCAN_WGSL: &str = include_str!("wgsl/tile_gen_scan_compute.wgsl");
pub const RADIX_SORT_COUNT_WGSL: &str = include_str!("wgsl/radix_sort_count.wgsl");
pub const RADIX_SORT_REDUCE_WGSL: &str = include_str!("wgsl/radix_sort_reduce.wgsl");
pub const RADIX_SORT_SCAN_WGSL: &str = include_str!("wgsl/radix_sort_scan.wgsl");
pub const RADIX_SORT_SCAN_ADD_WGSL: &str = include_str!("wgsl/radix_sort_scan_add.wgsl");
pub const RADIX_SORT_SCATTER_WGSL: &str = include_str!("wgsl/radix_sort_scatter.wgsl");
