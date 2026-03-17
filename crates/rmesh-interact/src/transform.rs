use glam::{Mat4, Quat, Vec3};

/// Spatial transform (position, rotation, scale).
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Kind of geometric primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    Cube,
    Sphere,
    Plane,
    Cylinder,
}

impl PrimitiveKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cube => "Cube",
            Self::Sphere => "Sphere",
            Self::Plane => "Plane",
            Self::Cylinder => "Cylinder",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Cube => 0,
            Self::Sphere => 1,
            Self::Plane => 2,
            Self::Cylinder => 3,
        }
    }
}

/// A named primitive with a transform.
#[derive(Debug, Clone)]
pub struct Primitive {
    pub kind: PrimitiveKind,
    pub transform: Transform,
    pub name: String,
}

impl Primitive {
    pub fn new(kind: PrimitiveKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            transform: Transform::default(),
            name: name.into(),
        }
    }
}
