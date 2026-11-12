//! Decisions and selection algorithms.
//!
//! Ported from solo-router's `router.py` (Apache 2.0, https://github.com/ddiddi/solo-router);
//! translated to Rust and modified by Armature AI Labs, 2026.
//! Rejection reasons and trace lines
//! are kept word for word so the two routers can be compared directly.

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::config::{ConfigError, Decision, FAILSAFE, ModelSpec, RouterConfig};
use crate::pyfmt::{as_f64, fmt_g, py_eq, py_round, truthy};
use crate::signals::{RobotRequest, SignalExtractor, Signals};

/// (quality, latency, cost) weights for the hybrid algorithm.
fn preference_weights(pref: &str) -> Option<(f64, f64, f64)> {
    match pref {
        "balanced" => Some((1.0, 0.3, 0.05)),
        "quality" => Some((1.0, 0.05, 0.02)),
        "latency" => Some((0.6, 1.0, 0.05)),
        "cost" => Some((0.6, 0.1, 1.0)),
        _ => None,
    }
}
const DEGRADED_CLOUD_PENALTY: f64 = 0.3;
const LOW_BATTERY_PENALTY_PER_W: f64 = 0.01;
const DEFAULT_LATENCY_SCALE_MS: f64 = 3000.0;
/// Observation streams that satisfy a model's "rgb" requirement.
const RGB_STREAMS: [&str; 2] = ["rgb", "wrist_rgb"];
const DEFAULT_LOCATIONS: [&str; 3] = ["onboard", "edge", "cloud"];

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub model: String,
    pub location: String,
    pub score: f64,
    pub p50_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteResult {
    pub decision: String,
    pub algorithm: String,
    pub model: String,
    pub location: String,
    pub base_url: Option<String>,
    pub api_format: Option<String>,
    pub provider: Option<String>,
    pub est_latency_ms: f64,
    pub signals: Signals,
    pub guardrails: Map<String, Value>,
    pub ranked: Vec<Candidate>,
    pub fallbacks: Vec<String>,
    pub rejected: IndexMap<String, String>,
    pub trace: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub stages: Vec<(String, RouteResult)>,
}

impl TaskResult {
    pub fn to_value(&self) -> Value {
        let stages: Vec<Value> = self
            .stages
            .iter()
            .map(|(name, route)| {
                let mut obj = Map::new();
                obj.insert("stage".into(), Value::String(name.clone()));
                if let Value::Object(fields) =
                    serde_json::to_value(route).expect("route serializes")
                {
                    obj.extend(fields);
                }
                Value::Object(obj)
            })
            .collect();
        json!({ "stages": stages })
    }
}

/// Does a decision's `when` clause match these signals? Supports `any` and `all` sub-clauses;
/// every other key names a signal and lists the values it may take.
pub fn matches(when: &Map<String, Value>, signals: &Map<String, Value>) -> bool {
    for (key, expected) in when {
        match key.as_str() {
            "any" => {
                let subs = expected.as_array().map(Vec::as_slice).unwrap_or_default();
                if !subs
                    .iter()
                    .any(|sub| sub.as_object().is_some_and(|s| matches(s, signals)))
                {
                    return false;
                }
            }
            "all" => {
                let subs = expected.as_array().map(Vec::as_slice).unwrap_or_default();
                if !subs
                    .iter()
                    .all(|sub| sub.as_object().is_some_and(|s| matches(s, signals)))
                {
                    return false;
                }
            }
            _ => {
                let actual = signals.get(key).unwrap_or(&Value::Null);
                let allowed: Vec<&Value> = match expected {
                    Value::Array(items) => items.iter().collect(),
                    other => vec![other],
                };
                if !allowed.iter().any(|a| py_eq(actual, a)) {
                    return false;
                }
            }
        }
    }
    true
}

pub struct Router {
    pub config: RouterConfig,
    extractor: SignalExtractor,
    min_elevated_tier: f64,
}

impl Router {
    pub fn new(config: RouterConfig) -> Result<Self, ConfigError> {
        let extractor = SignalExtractor::new(&config.signals)?;
        let min_elevated_tier = config
            .guardrails
            .get("elevated")
            .and_then(|e| e.get("min_safety_tier"))
            .and_then(as_f64)
            .unwrap_or(0.0);
        Ok(Self {
            config,
            extractor,
            min_elevated_tier,
        })
    }

    pub fn route(&self, request: &RobotRequest) -> RouteResult {
        let signals = self.extractor.extract(request);
        let signal_map = match serde_json::to_value(&signals).expect("signals serialize") {
            Value::Object(m) => m,
            _ => unreachable!(),
        };
        let mut trace = vec![format!(
            "signals: intent={} horizon={} safety={} latency={} network={}",
            signals.intent, signals.horizon, signals.safety, signals.latency_tier, signals.network
        )];

        let failsafe_decision = Decision {
            name: "failsafe".into(),
            priority: 0,
            when: Map::new(),
            candidates: vec![FAILSAFE.into()],
            algorithm: "static".into(),
            prefer: None,
        };
        let decision = self
            .config
            .decisions
            .iter()
            .find(|d| matches(&d.when, &signal_map))
            .unwrap_or(&failsafe_decision);
        trace.push(format!(
            "decision: {} (priority {}, {})",
            decision.name, decision.priority, decision.algorithm
        ));

        let mut feasible = Vec::new();
        let mut rejected = IndexMap::new();
        for name in &decision.candidates {
            let spec = &self.config.models[name];
            match self.reject_reason(spec, &signals, request) {
                Some(why) => {
                    trace.push(format!("reject {name}: {why}"));
                    rejected.insert(name.clone(), why);
                }
                None => feasible.push(spec),
            }
        }

        let ranked = self.rank(decision, &feasible, &signals);
        let chosen = match ranked.first() {
            Some(top) => {
                let chosen = &self.config.models[&top.model];
                trace.push(format!(
                    "select {} @ {} (score {:.3})",
                    chosen.name, chosen.location, top.score
                ));
                chosen
            }
            None => {
                trace.push(
                    "no feasible candidate: falling back to safety-controller (hold position)"
                        .into(),
                );
                &self.config.models[FAILSAFE]
            }
        };

        let mut fallbacks: Vec<String> = ranked.iter().skip(1).map(|c| c.model.clone()).collect();
        if chosen.name != FAILSAFE {
            fallbacks.push(FAILSAFE.into());
        }

        let guardrails = match signals.safety.as_str() {
            "elevated" => self
                .config
                .guardrails
                .get("elevated")
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter(|(k, _)| *k != "min_safety_tier")
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            "critical" => {
                let mut m = Map::new();
                m.insert("action".into(), Value::from("stop"));
                m.insert("hold_position".into(), Value::Bool(true));
                m
            }
            _ => Map::new(),
        };

        RouteResult {
            decision: decision.name.clone(),
            algorithm: decision.algorithm.clone(),
            model: chosen.name.clone(),
            location: chosen.location.clone(),
            base_url: chosen.base_url.clone(),
            api_format: chosen.api_format.clone(),
            provider: chosen.provider.clone(),
            est_latency_ms: chosen.p50_latency_ms,
            signals,
            guardrails,
            ranked,
            fallbacks,
            rejected,
            trace,
        }
    }

    /// Route each configured pipeline stage (e.g. perceive -> reason -> act) for one task.
    pub fn route_task(&self, request: &RobotRequest) -> TaskResult {
        let mut stages = Vec::new();
        for stage in &self.config.pipeline {
            let mut sub = request.clone();
            if let Some(forced) = stage.get("signals").and_then(Value::as_object) {
                for (k, v) in forced {
                    let text = v
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| v.to_string());
                    sub.overrides.insert(k.clone(), text);
                }
            }
            if !stage.get("control_loop").is_some_and(truthy) {
                sub.control_hz = None;
                sub.latency_budget_ms = None;
            }
            let name = stage
                .get("stage")
                .and_then(Value::as_str)
                .unwrap_or("stage")
                .to_string();
            stages.push((name, self.route(&sub)));
        }
        TaskResult { stages }
    }

    fn reject_reason(&self, spec: &ModelSpec, s: &Signals, req: &RobotRequest) -> Option<String> {
        if spec.name == FAILSAFE {
            return None;
        }
        let reachable = match &req.locations {
            Some(locations) => locations.contains(&spec.location),
            None => DEFAULT_LOCATIONS.contains(&spec.location.as_str()),
        };
        if !reachable {
            return Some(format!("{} not reachable", spec.location));
        }
        if let (Some(provider), Some(enabled)) = (&spec.provider, &req.providers) {
            if !enabled.contains(provider) {
                return Some(format!("no {provider} API key"));
            }
        }
        if !spec.runtimes.is_empty()
            && !spec
                .runtimes
                .iter()
                .any(|r| req.client_runtimes.contains(r))
        {
            return Some(format!("needs runtime {}", spec.runtimes.join(" or ")));
        }
        let family = s
            .embodiment
            .as_deref()
            .and_then(|e| self.config.embodiment_families.get(e))
            .map(String::as_str);
        if !spec.supports(s.embodiment.as_deref(), family) {
            let shown = s.embodiment.as_deref().unwrap_or("None");
            return Some(format!("does not support embodiment {shown}"));
        }
        for need in &spec.requires {
            let satisfied = if need == "rgb" {
                RGB_STREAMS
                    .iter()
                    .any(|stream| s.modalities.iter().any(|m| m == stream))
            } else {
                s.modalities.iter().any(|m| m == need)
            };
            if !satisfied {
                return Some(format!("needs {need} observation"));
            }
        }
        if s.network == "offline" && spec.location == "cloud" {
            return Some("robot is offline".into());
        }
        if let Some(budget_ms) = s.latency_budget_ms {
            let budget = budget_ms * f64::from(spec.action_chunk);
            if spec.p50_latency_ms > budget {
                return Some(if spec.action_chunk > 1 {
                    format!(
                        "p50 {}ms exceeds {}-step chunk budget {}ms",
                        fmt_g(spec.p50_latency_ms),
                        spec.action_chunk,
                        fmt_g(budget)
                    )
                } else {
                    format!(
                        "p50 {}ms exceeds budget {}ms",
                        fmt_g(spec.p50_latency_ms),
                        fmt_g(budget)
                    )
                });
            }
        }
        if let (Some(hz), Some(max_hz)) = (req.control_hz.filter(|hz| *hz != 0.0), spec.max_hz) {
            if max_hz < hz {
                return Some(format!(
                    "max {}Hz below control rate {}Hz",
                    fmt_g(max_hz),
                    fmt_g(hz)
                ));
            }
        }
        if s.safety == "elevated"
            && spec.role == "action"
            && (spec.safety_tier as f64) < self.min_elevated_tier
        {
            return Some(format!(
                "safety tier {} below required {}",
                spec.safety_tier,
                py_number(self.min_elevated_tier)
            ));
        }
        None
    }

    fn rank(&self, decision: &Decision, feasible: &[&ModelSpec], s: &Signals) -> Vec<Candidate> {
        let candidate = |m: &ModelSpec, score: f64| Candidate {
            model: m.name.clone(),
            location: m.location.clone(),
            score,
            p50_latency_ms: m.p50_latency_ms,
        };
        if feasible.is_empty() {
            return Vec::new();
        }
        match decision.algorithm.as_str() {
            "static" => return feasible.iter().map(|m| candidate(m, 1.0)).collect(),
            "latency_aware" => {
                let mut ordered = feasible.to_vec();
                ordered.sort_by(|a, b| {
                    b.quality
                        .total_cmp(&a.quality)
                        .then(a.p50_latency_ms.total_cmp(&b.p50_latency_ms))
                });
                return ordered.iter().map(|m| candidate(m, m.quality)).collect();
            }
            _ => {}
        }

        // A decision's own preference applies unless the request states one.
        let mut pref = if s.preference != "balanced" {
            s.preference.as_str()
        } else {
            decision
                .prefer
                .as_deref()
                .filter(|p| !p.is_empty())
                .unwrap_or("balanced")
        };
        if preference_weights(pref).is_none() {
            pref = "balanced";
        }
        let (wq, wl, wc) = preference_weights(pref).expect("known preference");
        let scale = s
            .latency_budget_ms
            .filter(|b| *b != 0.0)
            .unwrap_or(DEFAULT_LATENCY_SCALE_MS);
        let max_cost = feasible
            .iter()
            .map(|m| m.cost_per_1k)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_cost = if max_cost == 0.0 || !max_cost.is_finite() {
            1.0
        } else {
            max_cost
        };

        let mut scored: Vec<Candidate> = feasible
            .iter()
            .map(|m| {
                let mut score = wq * m.quality
                    - wl * (m.p50_latency_ms / scale).min(1.0)
                    - wc * m.cost_per_1k / max_cost;
                if s.network == "degraded" && m.location == "cloud" {
                    score -= DEGRADED_CLOUD_PENALTY;
                }
                if s.energy == "low" {
                    score -= LOW_BATTERY_PENALTY_PER_W * m.power_w;
                }
                candidate(m, py_round(score, 4))
            })
            .collect();
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored
    }
}

/// Python's `str()` of a number that may be an int or a float (e.g. `2` or `2.5`).
fn py_number(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DEFAULT_CONFIG_YAML;

    fn router() -> Router {
        Router::new(RouterConfig::from_yaml_str(DEFAULT_CONFIG_YAML).unwrap()).unwrap()
    }

    fn req(v: Value) -> RobotRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn stop_routes_to_failsafe_with_stop_guardrail() {
        let r = router().route(&req(json!({"instruction": "stop!", "embodiment": "so101"})));
        assert_eq!(r.signals.safety, "critical");
        assert_eq!(r.model, FAILSAFE);
        assert_eq!(r.guardrails.get("action"), Some(&Value::from("stop")));
    }

    #[test]
    fn every_route_ends_in_the_failsafe() {
        let r = router().route(&req(json!({"instruction": "pick up the red cup", "embodiment": "so101", "observations": ["rgb"], "control_hz": 30})));
        if r.model != FAILSAFE {
            assert_eq!(r.fallbacks.last().map(String::as_str), Some(FAILSAFE));
        }
    }

    #[test]
    fn offline_robots_never_get_cloud_models() {
        let r = router().route(&req(
            json!({"instruction": "why is joint 3 overheating", "network": "offline"}),
        ));
        assert_ne!(r.location, "cloud");
    }
}
