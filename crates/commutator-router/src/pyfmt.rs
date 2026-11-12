//! Small helpers that reproduce Python behaviour exactly, so Commutator makes the same
//! decisions and writes the same trace text as solo-router.

use serde_json::Value;

/// Python truthiness for a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `int(x)` for numbers and numeric strings.
pub fn as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

pub fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

/// Python `==` between JSON values: numbers compare by value (1 == 1.0, True == 1).
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (as_number(a), as_number(b)) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

fn as_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(_) | Value::Bool(_) => as_f64(v),
        _ => None,
    }
}

/// Python `round(x, ndigits)`: correctly rounded, ties to even on the exact binary value.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

/// Python `format(x, "g")`: 6 significant digits, trailing zeros removed,
/// scientific notation outside 1e-4 <= |x| < 1e6.
pub fn fmt_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    const PRECISION: i32 = 6;
    // Round to 6 significant digits first; the exponent of the rounded value decides the style.
    let sci = format!("{:.*e}", (PRECISION - 1) as usize, x);
    let (mantissa, exp) = sci.split_once('e').expect("scientific format");
    let exp: i32 = exp.parse().expect("exponent");
    if (-4..PRECISION).contains(&exp) {
        let decimals = (PRECISION - 1 - exp).max(0) as usize;
        strip_zeros(&format!("{x:.decimals$}"))
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip_zeros(mantissa), exp.abs())
    }
}

fn strip_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_g_matches_python() {
        // Expected values produced by Python's format(x, "g").
        let cases = [
            (50.0, "50"),
            (33.333333333333336, "33.3333"),
            (1000.0 / 30.0, "33.3333"),
            (0.5, "0.5"),
            (1500.0, "1500"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (999999.5, "1e+06"),
            (2.5, "2.5"),
            (100.0, "100"),
            (-12.5, "-12.5"),
        ];
        for (x, want) in cases {
            assert_eq!(fmt_g(x), want, "fmt_g({x})");
        }
    }

    #[test]
    fn py_round_ties_to_even_on_binary_value() {
        assert_eq!(py_round(0.12345, 4), 0.1235); // 0.12345 is slightly above the tie in binary
        assert_eq!(py_round(2.675, 2), 2.67); // famous Python example
        assert_eq!(py_round(-0.3, 4), -0.3);
    }

    #[test]
    fn python_equality_and_truthiness() {
        assert!(py_eq(&Value::from(1), &Value::from(1.0)));
        assert!(py_eq(&Value::Bool(true), &Value::from(1)));
        assert!(!py_eq(&Value::from("1"), &Value::from(1)));
        assert!(!truthy(&Value::from(0.0)));
        assert!(truthy(&Value::from("x")));
    }
}
