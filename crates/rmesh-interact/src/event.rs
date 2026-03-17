/// Platform-agnostic input events for the transform interaction system.
///
/// Both native (winit) and web (web-sys) viewers map their raw events
/// to these types before feeding them to [`TransformInteraction`].

/// Abstract key relevant to the interaction system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractKey {
    G,
    S,
    R,
    X,
    Y,
    Z,
    Shift,
    Enter,
    Escape,
    Backspace,
    Delete,
    Tab,
}

/// Mouse button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
}

/// Abstract input event for the interaction system.
#[derive(Debug, Clone)]
pub enum InteractEvent {
    KeyDown(InteractKey),
    KeyUp(InteractKey),
    /// Character input for numeric entry ('0'-'9', '.', '-').
    CharInput(char),
    MouseMove { dx: f32, dy: f32 },
    MouseDown { button: MouseButton },
    MouseUp { button: MouseButton },
}
