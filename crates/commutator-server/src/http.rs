//! HTTP API for Commutator.
//!
//! This first release routes only (dry run): it returns the routing decision and never calls a
//! model. Proxying to model backends, the scheduler and the safety filter come in later releases.

use std::{collections::HashMap, sync::Arc};

use axum::{
    Json, Router as HttpRouter,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use commutator_router::{RobotRequest, Router};
use serde_json::{Value, json};

pub const AUTO_MODEL: &str = "commutator/auto";

pub struct AppState {
    pub router: Router,
    pub dry_run: bool,
}

pub fn app(state: Arc<AppState>) -> HttpRouter {
    HttpRouter::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/providers", get(providers))
        .route("/v1/route", post(route))
        .route("/v1/task", post(task))
        .route("/v1/chat/completions", post(not_yet))
        .route("/v1/actions", post(not_yet))
        .fallback(not_found)
        .with_state(state)
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message.into() } })),
    )
        .into_response()
}

async fn not_found(uri: axum::http::Uri, method: axum::http::Method) -> Response {
    error(
        StatusCode::NOT_FOUND,
        format!("no route for {method} {}", uri.path()),
    )
}

async fn not_yet() -> Response {
    error(
        StatusCode::NOT_IMPLEMENTED,
        "this release only routes: calling model backends is not implemented yet. Use POST /v1/route or /v1/task to get the routing decision.",
    )
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "dry_run": state.dry_run,
        "models": state.router.config.models.len(),
    }))
}

/// Provider API keys available to this request: from the server's environment, or from
/// `x-<provider>-api-key` request headers.
fn keys(state: &AppState, headers: &HeaderMap) -> HashMap<String, String> {
    let mut keys = HashMap::new();
    for (name, provider) in &state.router.config.providers {
        if let Ok(key) = std::env::var(&provider.api_key_env) {
            if !key.is_empty() {
                keys.insert(name.clone(), key);
            }
        }
        if let Some(key) = headers
            .get(format!("x-{name}-api-key"))
            .and_then(|v| v.to_str().ok())
        {
            if !key.is_empty() {
                keys.insert(name.clone(), key.to_string());
            }
        }
    }
    keys
}

async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Json<Value> {
    let keys = keys(&state, &headers);
    let mut data = vec![json!({ "id": AUTO_MODEL, "object": "model", "owned_by": "commutator" })];
    for m in state.router.config.models.values() {
        let mut entry = json!({
            "id": m.name, "object": "model", "owned_by": m.provider.as_deref().unwrap_or(&m.location),
            "role": m.role, "kind": m.kind, "location": m.location, "provider": m.provider,
        });
        if let Some(p) = &m.provider {
            entry["configured"] = Value::Bool(keys.contains_key(p));
        }
        data.push(entry);
    }
    Json(json!({ "object": "list", "data": data }))
}

async fn providers(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Json<Value> {
    let keys = keys(&state, &headers);
    let config = &state.router.config;
    let data: Vec<Value> = config
        .providers
        .values()
        .map(|p| {
            json!({
                "id": p.name, "label": p.label, "base_url": p.base_url, "api_format": p.api_format,
                "api_key_env": p.api_key_env, "configured": keys.contains_key(&p.name),
                "models": config.models.values().filter(|m| m.provider.as_deref() == Some(&p.name)).map(|m| &m.name).collect::<Vec<_>>(),
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

/// A proxy can only call providers it has a key for, so outside a dry run it routes among those
/// unless the request narrows them itself. A dry run routes as if every provider were enabled.
fn prepare(state: &AppState, mut req: RobotRequest, headers: &HeaderMap) -> RobotRequest {
    if !state.dry_run {
        let keys = keys(state, headers);
        req.providers = Some(match req.providers.take() {
            None => state
                .router
                .config
                .providers
                .keys()
                .filter(|p| keys.contains_key(*p))
                .cloned()
                .collect(),
            Some(asked) => asked.into_iter().filter(|p| keys.contains_key(p)).collect(),
        });
    }
    req
}

/// Parse a robot request body. The error is the message for a 400 response.
fn parse(body: Result<Json<Value>, JsonRejection>) -> Result<RobotRequest, String> {
    let Json(body) = body.map_err(|e| e.body_text())?;
    serde_json::from_value(body).map_err(|e| format!("invalid robot request: {e}"))
}

fn header(name: &str, value: &str) -> Option<(HeaderName, HeaderValue)> {
    Some((
        HeaderName::from_bytes(name.as_bytes()).ok()?,
        HeaderValue::from_str(value).ok()?,
    ))
}

async fn route(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let req = match parse(body) {
        Ok(r) => prepare(&state, r, &headers),
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    let result = state.router.route(&req);
    let mut out = HeaderMap::new();
    for (name, value) in [
        ("x-commutator-decision", result.decision.as_str()),
        ("x-commutator-model", result.model.as_str()),
        ("x-commutator-location", result.location.as_str()),
        ("x-commutator-safety", result.signals.safety.as_str()),
    ] {
        if let Some((n, v)) = header(name, value) {
            out.insert(n, v);
        }
    }
    (out, Json(result)).into_response()
}

async fn task(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let req = match parse(body) {
        Ok(r) => prepare(&state, r, &headers),
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    let result = state.router.route_task(&req);
    let mut out = HeaderMap::new();
    for (stage, route) in &result.stages {
        if let Some((n, v)) = header(&format!("x-commutator-{stage}"), &route.model) {
            out.insert(n, v);
        }
    }
    (out, Json(result.to_value())).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use commutator_router::{RouterConfig, config::DEFAULT_CONFIG_YAML};
    use tower::ServiceExt;

    fn test_app() -> HttpRouter {
        let router =
            Router::new(RouterConfig::from_yaml_str(DEFAULT_CONFIG_YAML).unwrap()).unwrap();
        app(Arc::new(AppState {
            router,
            dry_run: true,
        }))
    }

    async fn call(req: Request<Body>) -> (StatusCode, HeaderMap, Value) {
        let res = test_app().oneshot(req).await.unwrap();
        let (status, headers) = (res.status(), res.headers().clone());
        let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (
            status,
            headers,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    fn post(path: &str, body: &str) -> Request<Body> {
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn health_is_ok() {
        let (status, _, body) = call(Request::get("/health").body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn route_stop_goes_to_failsafe() {
        let (status, headers, body) = call(post(
            "/v1/route",
            r#"{"instruction":"stop!","embodiment":"so101"}"#,
        ))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["model"], "safety-controller");
        assert_eq!(headers["x-commutator-safety"], "critical");
    }

    #[tokio::test]
    async fn task_returns_stages() {
        let (status, _, body) = call(post(
            "/v1/task",
            r#"{"instruction":"pick up the cup","embodiment":"so101","observations":["rgb"]}"#,
        ))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(!body["stages"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn bad_json_is_a_400_with_a_message() {
        let (status, _, body) = call(post("/v1/route", "{not json")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].as_str().unwrap().len() > 3);
    }

    #[tokio::test]
    async fn proxy_endpoints_say_not_implemented() {
        let (status, _, _) = call(post("/v1/chat/completions", "{}")).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    }
}
