/// Digit buffer for typing exact numeric values during modal transforms.
///
/// Accepts '0'-'9', '.' (once), '-' (toggles sign at start).
/// Used e.g. `G X 3 . 5 Enter` = move +3.5 on X.
#[derive(Debug, Clone, Default)]
pub struct NumericInput {
    buf: String,
}

impl NumericInput {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a character. Returns `true` if the character was accepted.
    pub fn push(&mut self, ch: char) -> bool {
        match ch {
            '0'..='9' => {
                self.buf.push(ch);
                true
            }
            '.' => {
                if !self.buf.contains('.') {
                    self.buf.push('.');
                    true
                } else {
                    false
                }
            }
            '-' => {
                // Toggle sign
                if self.buf.starts_with('-') {
                    self.buf.remove(0);
                } else {
                    self.buf.insert(0, '-');
                }
                true
            }
            _ => false,
        }
    }

    /// Remove the last character (backspace).
    pub fn backspace(&mut self) {
        self.buf.pop();
    }

    /// Parse the current buffer as f32. Returns `None` if empty or invalid.
    pub fn value(&self) -> Option<f32> {
        if self.buf.is_empty() || self.buf == "-" || self.buf == "." || self.buf == "-." {
            return None;
        }
        self.buf.parse::<f32>().ok()
    }

    /// Display string for HUD overlay.
    pub fn display(&self) -> &str {
        &self.buf
    }

    /// Whether the buffer is empty (no digits entered).
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_digits() {
        let mut n = NumericInput::new();
        assert!(n.push('3'));
        assert!(n.push('.'));
        assert!(n.push('5'));
        assert_eq!(n.value(), Some(3.5));
        assert_eq!(n.display(), "3.5");
    }

    #[test]
    fn test_negative() {
        let mut n = NumericInput::new();
        assert!(n.push('-'));
        assert!(n.push('7'));
        assert_eq!(n.value(), Some(-7.0));

        // Toggle sign off
        assert!(n.push('-'));
        assert_eq!(n.value(), Some(7.0));
    }

    #[test]
    fn test_no_double_dot() {
        let mut n = NumericInput::new();
        assert!(n.push('1'));
        assert!(n.push('.'));
        assert!(!n.push('.')); // rejected
        assert!(n.push('2'));
        assert_eq!(n.value(), Some(1.2));
    }

    #[test]
    fn test_backspace() {
        let mut n = NumericInput::new();
        n.push('4');
        n.push('2');
        n.backspace();
        assert_eq!(n.value(), Some(4.0));
        n.backspace();
        assert!(n.value().is_none());
        assert!(n.is_empty());
    }

    #[test]
    fn test_empty_and_partial() {
        let n = NumericInput::new();
        assert!(n.value().is_none());
        assert!(n.is_empty());

        let mut n2 = NumericInput::new();
        n2.push('-');
        assert!(n2.value().is_none()); // just "-" is not a valid number

        let mut n3 = NumericInput::new();
        n3.push('.');
        assert!(n3.value().is_none()); // just "." is not valid
    }

    #[test]
    fn test_integer() {
        let mut n = NumericInput::new();
        n.push('1');
        n.push('0');
        assert_eq!(n.value(), Some(10.0));
    }
}
