//! Load and validate router configuration.
//!
//! Ported from solo-router's `config.py` (Apache 2.0, https://github.com/ddiddi/solo-router);
//! translated to Rust and modified by Armature AI Labs, 2026.
//! The YAML format is the same, so a
//! solo-router config file loads unchanged.

use std::path::Path;

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};

/// "browser" models run client-side through ONNX Runtime Web (WebGPU, or WASM as a fallback).
pub const LOCATIONS: [&str; 4] = ["onboard", "edge", "cloud", "browser"];
pub const ROLES: [&str; 4] = ["perception", "reasoning", "action", "safety"];
pub const ALGORITHMS: [&str; 3] = ["static", "latency_aware", "hybrid"];
pub const API_FORMATS: [&str; 2] = ["openai", "anthropic"];

/// The deterministic fail-safe every route ends in. A config must define it.
pub const FAILSAFE: &str = "safety-controller";

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{0}")]
    Invalid(String),
    #[error("could not read config {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("could not parse config: {0}")]
    Parse(String),
}

fn invalid(msg: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(msg.into())
}

fn default_embodiments() -> Vec<String> {
    vec!["*".to_string()]
}
fn default_one() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub name: String,
    pub kind: String,
    pub role: String,
    pub location: String,
    pub p50_latency_ms: f64,
    pub quality: f64,
    #[serde(default)]
    pub max_hz: Option<f64>,
    /// Steps of motion predicted per inference call. A chunked policy only has to answer
    /// once per chunk, so its latency budget is chunk * (1000 / control_hz).
    #[serde(default = "default_one")]
    pub action_chunk: u32,
    #[serde(default)]
    pub cost_per_1k: f64,
    #[serde(default)]
    pub power_w: f64,
    #[serde(default = "default_embodiments")]
    pub embodiments: Vec<String>,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default = "default_safety_tier")]
    pub safety_tier: i64,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_format: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Cloud provider that serves this model (a key of `providers`).
    #[serde(default)]
    pub provider: Option<String>,
    /// Model ID the provider expects, when it differs from `name`.
    #[serde(default)]
    pub upstream_model: Option<String>,
    /// Client runtimes the model needs, any one of which suffices (e.g. webgpu, wasm).
    #[serde(default)]
    pub runtimes: Vec<String>,
    /// How a browser loads it, for transformers.js.
    #[serde(default)]
    pub onnx: Option<Value>,
    #[serde(default)]
    pub description: String,
}

fn default_safety_tier() -> i64 {
    1
}

impl ModelSpec {
    pub fn supports(&self, embodiment: Option<&str>, family: Option<&str>) -> bool {
        let Some(embodiment) = embodiment else {
            return true;
        };
        if self.embodiments.iter().any(|e| e == "*") {
            return true;
        }
        self.embodiments.iter().any(|e| e == embodiment)
            || family.is_some_and(|f| self.embodiments.iter().any(|e| e == f))
    }

    /// The model ID to send upstream: `upstream_model`, or `name` without its "provider/" prefix.
    pub fn upstream_id(&self) -> &str {
        if let Some(upstream) = &self.upstream_model {
            return upstream;
        }
        if let Some(provider) = &self.provider {
            let prefix = format!("{provider}/");
            if let Some(rest) = self.name.strip_prefix(&prefix) {
                return rest;
            }
        }
        &self.name
    }
}

/// A hosted model API you bring your own key for (OpenAI, Anthropic, Nebius Token Factory, ...).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSpec {
    #[serde(skip)]
    pub name: String,
    pub label: String,
    pub base_url: String,
    pub api_format: String,
    pub api_key_env: String,
    #[serde(default)]
    pub console_url: String,
}

#[derive(Debug, Clone)]
pub struct Decision {
    pub name: String,
    pub priority: i64,
    pub when: Map<String, Value>,
    pub candidates: Vec<String>,
    pub algorithm: String,
    pub prefer: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub signals: Value,
    pub models: IndexMap<String, ModelSpec>,
    /// Sorted by priority, highest first (stable for equal priorities).
    pub decisions: Vec<Decision>,
    pub guardrails: Value,
    pub listener: Value,
    pub raw: Value,
    /// Specific robots mapped to the embodiment family policies are trained for, e.g. so101 -> arm.
    pub embodiment_families: IndexMap<String, String>,
    /// Ordered stages for /v1/task: each forces some signals and routes independently.
    pub pipeline: Vec<Value>,
    pub providers: IndexMap<String, ProviderSpec>,
}

/// `data.get(key) or default`: missing, null, false, 0, "" and empty containers are all falsy in Python.
fn get_truthy<'a>(data: &'a Value, key: &str) -> Option<&'a Value> {
    data.get(key).filter(|v| crate::pyfmt::truthy(v))
}

impl RouterConfig {
    pub fn from_value(data: Value) -> Result<Self, ConfigError> {
        if !data.is_object() {
            return Err(invalid("config must be a mapping"));
        }

        let mut providers = IndexMap::new();
        if let Some(entries) = get_truthy(&data, "providers").and_then(Value::as_object) {
            for (name, entry) in entries {
                let mut spec: ProviderSpec = serde_json::from_value(entry.clone())
                    .map_err(|e| invalid(format!("provider {name}: {e}")))?;
                spec.name = name.clone();
                if !API_FORMATS.contains(&spec.api_format.as_str()) {
                    return Err(invalid(format!(
                        "provider {name}: unknown api_format {:?}",
                        spec.api_format
                    )));
                }
                providers.insert(name.clone(), spec);
            }
        }

        let mut models: IndexMap<String, ModelSpec> = IndexMap::new();
        for entry in data
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let label = entry
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string();
            let mut spec: ModelSpec = serde_json::from_value(entry.clone())
                .map_err(|e| invalid(format!("model {label}: {e}")))?;
            if !LOCATIONS.contains(&spec.location.as_str()) {
                return Err(invalid(format!(
                    "model {}: unknown location {:?}",
                    spec.name, spec.location
                )));
            }
            if !ROLES.contains(&spec.role.as_str()) {
                return Err(invalid(format!(
                    "model {}: unknown role {:?}",
                    spec.name, spec.role
                )));
            }
            if models.contains_key(&spec.name) {
                return Err(invalid(format!("duplicate model {}", spec.name)));
            }
            if let Some(provider_name) = spec.provider.clone() {
                let provider = providers.get(&provider_name).ok_or_else(|| {
                    invalid(format!(
                        "model {}: unknown provider {provider_name:?}",
                        spec.name
                    ))
                })?;
                spec.base_url = spec.base_url.or_else(|| Some(provider.base_url.clone()));
                spec.api_format = spec
                    .api_format
                    .or_else(|| Some(provider.api_format.clone()));
                spec.api_key_env = spec
                    .api_key_env
                    .or_else(|| Some(provider.api_key_env.clone()));
            }
            models.insert(spec.name.clone(), spec);
        }

        let mut decisions = Vec::new();
        for entry in data
            .get("decisions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let name = entry
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("decision without a name"))?
                .to_string();
            let priority = match entry.get("priority") {
                None | Some(Value::Null) => 0,
                Some(v) => crate::pyfmt::as_int(v).ok_or_else(|| {
                    invalid(format!("decision {name}: priority must be a number"))
                })?,
            };
            let when = get_truthy(entry, "when")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let candidates: Vec<String> = entry
                .get("candidates")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid(format!("decision {name}: candidates are required")))?
                .iter()
                .map(|c| {
                    c.as_str().map(str::to_string).ok_or_else(|| {
                        invalid(format!("decision {name}: candidates must be names"))
                    })
                })
                .collect::<Result<_, _>>()?;
            let algorithm = entry
                .get("algorithm")
                .and_then(Value::as_str)
                .unwrap_or("static")
                .to_string();
            let prefer = entry
                .get("prefer")
                .and_then(Value::as_str)
                .map(str::to_string);
            if !ALGORITHMS.contains(&algorithm.as_str()) {
                return Err(invalid(format!(
                    "decision {name}: unknown algorithm {algorithm:?}"
                )));
            }
            let missing: Vec<&String> = candidates
                .iter()
                .filter(|c| !models.contains_key(*c))
                .collect();
            if !missing.is_empty() {
                return Err(invalid(format!(
                    "decision {name}: unknown candidates {missing:?}"
                )));
            }
            decisions.push(Decision {
                name,
                priority,
                when,
                candidates,
                algorithm,
                prefer,
            });
        }
        decisions.sort_by(|a, b| b.priority.cmp(&a.priority));

        if !models.contains_key(FAILSAFE) {
            return Err(invalid(format!(
                "a model named '{FAILSAFE}' is required as the fail-safe"
            )));
        }

        let embodiment_families = data
            .get("embodiment_families")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            signals: data
                .get("signals")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
            models,
            decisions,
            guardrails: data
                .get("guardrails")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
            listener: data
                .get("listener")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
            embodiment_families,
            pipeline: data
                .get("pipeline")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            providers,
            raw: data,
        })
    }

    pub fn from_yaml_str(text: &str) -> Result<Self, ConfigError> {
        let data: Value =
            serde_yaml_ng::from_str(text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        Self::from_value(data)
    }

    /// Load a `.json` or YAML config file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if path.extension().is_some_and(|e| e == "json") {
            let data: Value =
                serde_json::from_str(&text).map_err(|e| ConfigError::Parse(e.to_string()))?;
            Self::from_value(data)
        } else {
            Self::from_yaml_str(&text)
        }
    }
}

/// The default config shipped with Commutator (solo-router's model set).
pub const DEFAULT_CONFIG_YAML: &str = include_str!("../router.yaml");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_loads() {
        let cfg = RouterConfig::from_yaml_str(DEFAULT_CONFIG_YAML).unwrap();
        assert!(cfg.models.contains_key(FAILSAFE));
        assert!(!cfg.decisions.is_empty());
        assert!(
            cfg.decisions
                .windows(2)
                .all(|w| w[0].priority >= w[1].priority)
        );
    }

    #[test]
    fn rejects_missing_failsafe() {
        let err = RouterConfig::from_yaml_str("models: []").unwrap_err();
        assert!(err.to_string().contains("safety-controller"));
    }

    #[test]
    fn rejects_unknown_model_fields() {
        let yaml = "models:\n  - {name: a, kind: x, role: action, location: onboard, p50_latency_ms: 1, quality: 1, typo: 3}\n";
        assert!(
            RouterConfig::from_yaml_str(yaml)
                .unwrap_err()
                .to_string()
                .contains("model a")
        );
    }
}
