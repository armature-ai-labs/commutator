//! Parity with solo-router: replay every request in tests/parity/fixtures.json (produced by
//! running solo-router's Python, see scripts/gen_parity_fixtures.py) and require the same
//! result, field by field. Numbers compare by value with a 1e-9 tolerance (Python keeps
//! integers as ints, Rust uses f64); everything else must match exactly, including trace text.

use std::path::PathBuf;

use commutator_router::{RobotRequest, Router, RouterConfig, config::DEFAULT_CONFIG_YAML};
use serde_json::Value;

fn fixtures() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/fixtures.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid fixtures JSON")
}

fn diff(path: &str, expected: &Value, actual: &Value, out: &mut Vec<String>) {
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            if (a - b).abs() > 1e-9 {
                out.push(format!("{path}: expected {a}, got {b}"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            for key in a.keys().chain(b.keys().filter(|k| !a.contains_key(*k))) {
                let (ea, eb) = (
                    a.get(key).unwrap_or(&Value::Null),
                    b.get(key).unwrap_or(&Value::Null),
                );
                if !a.contains_key(key) || !b.contains_key(key) {
                    out.push(format!(
                        "{path}.{key}: present on one side only (expected {ea}, got {eb})"
                    ));
                } else {
                    diff(&format!("{path}.{key}"), ea, eb, out);
                }
            }
            // Python dicts and our maps both keep insertion order; check it too.
            let (ka, kb): (Vec<_>, Vec<_>) = (a.keys().collect(), b.keys().collect());
            if ka != kb && ka.len() == kb.len() {
                out.push(format!(
                    "{path}: key order differs: expected {ka:?}, got {kb:?}"
                ));
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                out.push(format!(
                    "{path}: expected {} items, got {}: {expected} vs {actual}",
                    a.len(),
                    b.len()
                ));
                return;
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                diff(&format!("{path}[{i}]"), x, y, out);
            }
        }
        _ if expected == actual => {}
        _ => out.push(format!("{path}: expected {expected}, got {actual}")),
    }
}

fn router() -> Router {
    Router::new(RouterConfig::from_yaml_str(DEFAULT_CONFIG_YAML).unwrap()).unwrap()
}

fn report(kind: &str, total: usize, failures: Vec<String>) {
    if !failures.is_empty() {
        let shown: Vec<_> = failures.iter().take(15).cloned().collect();
        panic!(
            "{} of {total} {kind} differ from solo-router. First differences:\n{}",
            failures.len(),
            shown.join("\n")
        );
    }
}

#[test]
fn routes_match_solo_router() {
    let fixtures = fixtures();
    let router = router();
    let cases = fixtures["routes"].as_array().expect("routes");
    assert!(cases.len() > 900, "expected the full parity grid");
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let request: RobotRequest =
            serde_json::from_value(case["request"].clone()).expect("request parses");
        let actual = serde_json::to_value(router.route(&request)).unwrap();
        let mut diffs = Vec::new();
        diff("route", &case["expected"], &actual, &mut diffs);
        if !diffs.is_empty() {
            failures.push(format!(
                "case {i} {}: {}",
                case["request"],
                diffs.join("; ")
            ));
        }
    }
    report("routes", cases.len(), failures);
}

#[test]
fn tasks_match_solo_router() {
    let fixtures = fixtures();
    let router = router();
    let cases = fixtures["tasks"].as_array().expect("tasks");
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let request: RobotRequest =
            serde_json::from_value(case["request"].clone()).expect("request parses");
        let actual = router.route_task(&request).to_value();
        let mut diffs = Vec::new();
        diff("task", &case["expected"], &actual, &mut diffs);
        if !diffs.is_empty() {
            failures.push(format!(
                "task {i} {}: {}",
                case["request"],
                diffs.join("; ")
            ));
        }
    }
    report("tasks", cases.len(), failures);
}
