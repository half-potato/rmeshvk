//! rmesh-compositor: Opaque primitive rendering + depth compositing with tet volume rendering.

pub mod geometry;
pub mod primitive_pass;
pub mod compositor_pass;

pub use geometry::{PrimitiveGeometry, PrimitiveVertex};
pub use primitive_pass::{MrtViews, PrimitivePipeline, PrimitiveTargets, record_primitive_pass};
pub use compositor_pass::{
    CompositorPipeline, CompositorTargets, CompositorUniforms,
    create_compositor_bind_group, record_composite,
};
