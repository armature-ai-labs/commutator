//! # commutator-router
//!
//! The routing core of [Commutator](https://github.com/armature-ai-labs/commutator), an
//! open-source AI gateway for robots. It turns a robot request into signals (intent, horizon,
//! safety, latency tier, network, energy), picks the first matching decision by priority, rejects
//! models that can't serve the request (location, provider keys, runtimes, embodiment, required
//! observations, latency budget, control rate, safety tier), ranks the rest, and always ends the
//! fallback chain in the deterministic `safety-controller`.
//!
//! This crate is a Rust port of [solo-router](https://github.com/ddiddi/solo-router) by
//! Dhruv Diddi (Apache 2.0), which adapts the signal, decision, algorithm architecture of
//! [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). It reads the same
//! YAML config and makes the same decisions; a parity test checks this on hundreds of requests.
//!
//! ```
//! use commutator_router::{Router, RouterConfig, RobotRequest, config::DEFAULT_CONFIG_YAML};
//!
//! let router = Router::new(RouterConfig::from_yaml_str(DEFAULT_CONFIG_YAML).unwrap()).unwrap();
//! let request: RobotRequest = serde_json::from_str(r#"{"instruction": "stop!"}"#).unwrap();
//! let route = router.route(&request);
//! assert_eq!(route.model, "safety-controller");
//! ```

pub mod config;
pub mod pyfmt;
pub mod router;
pub mod signals;

pub use config::{ConfigError, FAILSAFE, ModelSpec, ProviderSpec, RouterConfig};
pub use router::{Candidate, RouteResult, Router, TaskResult};
pub use signals::{RobotRequest, Signals};
