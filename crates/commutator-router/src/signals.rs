//! Turn a robot request into routing signals.
//!
//! Ported from solo-router's `signals.py` (Apache 2.0, https://github.com/ddiddi/solo-router);
//! translated to Rust and modified by Armature AI Labs, 2026.

use fancy_regex::Regex;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::ConfigError;
use crate::pyfmt::as_f64;

fn default_network() -> String {
    "online".into()
}
fn default_preference() -> String {
    "balanced".into()
}
fn default_model() -> String {
    "commutator/auto".into()
}

/// What a robot (or an agent acting for it) is asking for, plus the constraints it is under.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RobotRequest {
    pub instruction: String,
    pub embodiment: Option<String>,
    /// Observation streams that accompany the request: rgb, depth, wrist_rgb, proprio, lidar, audio, force.
    pub observations: Vec<String>,
    pub control_hz: Option<f64>,
    pub latency_budget_ms: Option<f64>,
    /// online | degraded | offline
    #[serde(default = "default_network")]
    pub network: String,
    pub battery_pct: Option<f64>,
    pub humans_nearby: bool,
    pub estop: bool,
    /// quality | latency | cost | balanced
    #[serde(default = "default_preference")]
    pub preference: String,
    #[serde(default = "default_model")]
    pub model: String,
    /// Where this client can reach compute. None means onboard, edge and cloud.
    pub locations: Option<Vec<String>>,
    pub client_runtimes: Vec<String>,
    /// Cloud providers the caller has API keys for. None means every configured provider.
    pub providers: Option<Vec<String>>,
    /// Signals an orchestrator already knows. Only intent and horizon can be overridden;
    /// safety is always computed from the request.
    pub overrides: IndexMap<String, String>,
}

impl RobotRequest {
    /// Build a request from an OpenAI chat body plus an optional `robot` extension object.
    pub fn from_openai(body: &Value) -> Result<Self, serde_json::Error> {
        let mut text = String::new();
        let mut has_image = false;
        if let Some(messages) = body.get("messages").and_then(Value::as_array) {
            if let Some(msg) = messages
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            {
                match msg.get("content") {
                    Some(Value::String(s)) => text = s.clone(),
                    Some(Value::Array(parts)) => {
                        let mut texts = Vec::new();
                        for part in parts {
                            match part.get("type").and_then(Value::as_str) {
                                Some("text") => texts.push(
                                    part.get("text")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string(),
                                ),
                                Some("image_url" | "input_image" | "image") => has_image = true,
                                _ => {}
                            }
                        }
                        text = texts.join(" ");
                    }
                    _ => {}
                }
            }
        }
        let robot = body
            .get("robot")
            .filter(|r| r.is_object())
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));
        let mut req: RobotRequest = serde_json::from_value(robot)?;
        if req.instruction.is_empty() {
            req.instruction = text;
        }
        if let Some(model) = body.get("model").and_then(Value::as_str) {
            req.model = model.to_string();
        }
        if has_image && !req.observations.iter().any(|o| o == "rgb") {
            req.observations.push("rgb".into());
        }
        Ok(req)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Signals {
    pub intent: String,
    pub intent_scores: IndexMap<String, i64>,
    /// short | long
    pub horizon: String,
    /// normal | elevated | critical
    pub safety: String,
    pub safety_reasons: Vec<String>,
    pub latency_budget_ms: Option<f64>,
    /// realtime | interactive | deliberative
    pub latency_tier: String,
    pub modalities: Vec<String>,
    pub embodiment: Option<String>,
    pub network: String,
    /// normal | low
    pub energy: String,
    pub preference: String,
}

/// A whole-word phrase match on lowercased text, equivalent to the Python regex
/// `(?<![a-z0-9])PHRASE(?![a-z0-9])`.
#[derive(Debug, Clone)]
struct Phrase {
    /// The phrase as written in the config (reported in safety reasons).
    original: String,
    chars: Vec<char>,
}

fn is_word(c: char) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit()
}

impl Phrase {
    fn new(phrase: &str) -> Self {
        Self {
            chars: phrase.to_lowercase().chars().collect(),
            original: phrase.to_string(),
        }
    }

    /// Number of non-overlapping matches, scanning left to right like `re.findall`.
    fn count(&self, haystack: &[char]) -> usize {
        let n = self.chars.len();
        if n == 0 {
            // An empty phrase matches at every position whose neighbours are not word characters.
            return (0..=haystack.len())
                .filter(|&i| {
                    (i == 0 || !is_word(haystack[i - 1]))
                        && (i == haystack.len() || !is_word(haystack[i]))
                })
                .count();
        }
        let (mut i, mut hits) = (0, 0);
        while i + n <= haystack.len() {
            if haystack[i..i + n] == self.chars[..]
                && (i == 0 || !is_word(haystack[i - 1]))
                && (i + n == haystack.len() || !is_word(haystack[i + n]))
            {
                hits += 1;
                i += n;
            } else {
                i += 1;
            }
        }
        hits
    }

    fn found(&self, haystack: &[char]) -> bool {
        self.count(haystack) > 0
    }
}

pub struct SignalExtractor {
    intents: IndexMap<String, Vec<Phrase>>,
    long_markers: f64,
    long_words: f64,
    markers: Vec<Phrase>,
    critical: Vec<Regex>,
    hazards: Vec<Phrase>,
    realtime_ms: f64,
    interactive_ms: f64,
    low_battery: f64,
}

fn num(v: Option<&Value>, default: f64) -> f64 {
    v.and_then(as_f64).unwrap_or(default)
}

fn phrases(v: Option<&Value>) -> Vec<Phrase> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().map(|w| Phrase::new(&value_text(w))).collect())
        .unwrap_or_default()
}

/// Python `str(x)` for the scalar values YAML lexicons hold (e.g. `on`, `1`).
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

impl SignalExtractor {
    pub fn new(cfg: &Value) -> Result<Self, ConfigError> {
        let intents = cfg
            .get("intent")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(name, words)| (name.clone(), phrases(Some(words))))
                    .collect()
            })
            .unwrap_or_default();
        let horizon = cfg.get("horizon");
        let safety = cfg.get("safety");
        let latency = cfg.get("latency");
        let critical = safety
            .and_then(|s| s.get("critical_patterns"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|p| {
                let pattern = value_text(p);
                Regex::new(&format!("(?i){pattern}")).map_err(|e| {
                    ConfigError::Invalid(format!(
                        "signals.safety.critical_patterns: {pattern:?}: {e}"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            intents,
            long_markers: num(horizon.and_then(|h| h.get("long_markers")), 2.0),
            long_words: num(horizon.and_then(|h| h.get("long_words")), 24.0),
            markers: phrases(horizon.and_then(|h| h.get("markers"))),
            critical,
            hazards: phrases(safety.and_then(|s| s.get("hazard_words"))),
            realtime_ms: num(latency.and_then(|l| l.get("realtime_ms")), 50.0),
            interactive_ms: num(latency.and_then(|l| l.get("interactive_ms")), 500.0),
            low_battery: num(
                cfg.get("energy").and_then(|e| e.get("low_battery_pct")),
                20.0,
            ),
        })
    }

    pub fn extract(&self, req: &RobotRequest) -> Signals {
        let text = req.instruction.to_lowercase().trim().to_string();
        let chars: Vec<char> = text.chars().collect();

        let scores: IndexMap<String, i64> = self
            .intents
            .iter()
            .map(|(name, kws)| {
                (
                    name.clone(),
                    kws.iter().filter(|kw| kw.found(&chars)).count() as i64,
                )
            })
            .collect();
        let best = scores.values().copied().max().unwrap_or(0);
        let mut intent = if best > 0 {
            scores
                .iter()
                .find(|(_, s)| **s == best)
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| "conversation".into())
        } else {
            "conversation".into()
        };

        let marker_hits =
            self.markers.iter().map(|m| m.count(&chars)).sum::<usize>() + text.matches(',').count();
        let long = marker_hits as f64 >= self.long_markers
            || text.split_whitespace().count() as f64 >= self.long_words;
        let mut horizon = if long { "long" } else { "short" }.to_string();

        if let Some(v) = req.overrides.get("intent") {
            intent = v.clone();
        }
        if let Some(v) = req.overrides.get("horizon") {
            horizon = v.clone();
        }

        let (mut safety, mut reasons) = ("normal".to_string(), Vec::new());
        if req.estop {
            safety = "critical".into();
            reasons = vec!["estop flag set".to_string()];
        } else if self
            .critical
            .iter()
            .any(|rx| rx.is_match(&text).unwrap_or(false))
        {
            safety = "critical".into();
            reasons = vec!["stop command in instruction".to_string()];
        } else {
            let hazards: Vec<&str> = self
                .hazards
                .iter()
                .filter(|h| h.found(&chars))
                .map(|h| h.original.as_str())
                .collect();
            if !hazards.is_empty() {
                reasons.push(format!("hazard words: {}", hazards.join(", ")));
            }
            if req.humans_nearby && (intent == "manipulation" || intent == "navigation") {
                reasons.push(format!("humans nearby during {intent}"));
            }
            if !reasons.is_empty() {
                safety = "elevated".into();
            }
        }

        let mut budget = req.latency_budget_ms;
        if budget.is_none() {
            if let Some(hz) = req.control_hz.filter(|hz| *hz != 0.0) {
                budget = Some(1000.0 / hz);
            }
        }
        let latency_tier = match budget {
            Some(b) if b < self.realtime_ms => "realtime",
            Some(b) if b < self.interactive_ms => "interactive",
            _ => "deliberative",
        }
        .to_string();

        let energy = if req.battery_pct.is_some_and(|b| b < self.low_battery) {
            "low"
        } else {
            "normal"
        }
        .to_string();

        let modalities = if text.is_empty() {
            req.observations.clone()
        } else {
            std::iter::once("text".to_string())
                .chain(req.observations.iter().cloned())
                .collect()
        };

        Signals {
            intent,
            intent_scores: scores,
            horizon,
            safety,
            safety_reasons: reasons,
            latency_budget_ms: budget,
            latency_tier,
            modalities,
            embodiment: req.embodiment.clone(),
            network: req.network.clone(),
            energy,
            preference: req.preference.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn phrase_matches_whole_words_only() {
        assert!(Phrase::new("pick up").found(&chars("please pick up the cup")));
        assert!(!Phrase::new("cup").found(&chars("cupboard")));
        assert!(!Phrase::new("hot").found(&chars("photo")));
        assert!(Phrase::new("hot").found(&chars("it's hot!")));
        assert_eq!(Phrase::new("then").count(&chars("then go, then stop")), 2);
    }

    #[test]
    fn non_overlapping_counts_like_findall() {
        assert_eq!(Phrase::new("aa").count(&chars("aa aa aaa")), 2);
    }
}
