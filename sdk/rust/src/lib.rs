//! Thin typed client for the Decision-Maker service (`/v1/decisionmaker`).
//!
//! Mirrors the wire contract in `spec/openapi.yaml`: Choice / Boolean / Score
//! questions in, per-candidate probability distributions out, with
//! spread-based confidence and typed errors. No `unsafe`.
//!
//! ## Features
//! - `async` (default): [`Client`] backed by `reqwest` + `tokio`.
//! - `blocking`: a [`blocking::Client`] in a submodule.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[cfg(feature = "blocking")]
pub mod blocking;

// ---------------------------------------------------------------------------
// Identifiers / state (structured, not only text)
// ---------------------------------------------------------------------------

/// A structured value: string, object, array, or null. `State` and question
/// fields accept this so the model sees labeled JSON rather than a flattened
/// template.
pub type Structured = serde_json::Value;

// ---------------------------------------------------------------------------
// Questions
// ---------------------------------------------------------------------------

/// A typed question. `instructions` is optional on all three types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooleanQuestion {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Structured>,
    /// Optional `{true, false}` descriptions of what a yes / a no mean.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub criteria: Option<BooleanCriteria>,
}

/// `{true, false}` descriptions; both optional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BooleanCriteria {
    #[serde(skip_serializing_if = "Option::is_none", rename = "true")]
    pub true_: Option<Structured>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "false")]
    pub false_: Option<Structured>,
}

/// An option question. Invariant: `2..=255` options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestion {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Structured>,
    /// option-key -> description (or null). Serialized as a JSON object
    /// (2..=255 keys), matching the server's Choice criteria schema.
    pub criteria: HashMap<String, Option<Structured>>,
}

impl ChoiceQuestion {
    /// Constructs a choice question, enforcing the 2..=255 option invariant.
    pub fn new(
        instructions: impl Into<Option<Structured>>,
        criteria: HashMap<String, Option<Structured>>,
    ) -> Result<Self, Error> {
        let n = criteria.len();
        if !(2..=255).contains(&n) {
            return Err(Error::Usage(format!(
                "choice criteria must have 2..=255 options (got {n})"
            )));
        }
        Ok(Self {
            instructions: instructions.into(),
            criteria,
        })
    }
}

/// An ordinal question. Invariant: `2..=10` levels, Low-to-High.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreQuestion {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Structured>,
    /// Ordered level descriptions, LOW to HIGH.
    pub criteria: Vec<Structured>,
}

impl ScoreQuestion {
    /// Constructs a score question, enforcing the 2..=10 level invariant.
    pub fn new(
        instructions: impl Into<Option<Structured>>,
        criteria: Vec<Structured>,
    ) -> Result<Self, Error> {
        let n = criteria.len();
        if !(2..=10).contains(&n) {
            return Err(Error::Usage(format!(
                "score criteria must have 2..=10 levels (got {n})"
            )));
        }
        Ok(Self {
            instructions: instructions.into(),
            criteria,
        })
    }
}

/// One of the three question types. Internally tagged as `type` on the wire
/// (e.g. `{"type":"boolean", ...}`), matching the OpenAPI discriminator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Boolean(BooleanQuestion),
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
}

/// The request body for `/v1/decisionmaker`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// A string, object, or array describing the situation.
    pub state: State,
    /// Optional pinned-model selector; the server defaults if omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// qid -> question. qids are never sent to the model; answers come back
    /// under the same keys.
    pub questions: HashMap<String, Question>,
}

/// The `state` can be a non-empty string, an object, or a non-empty array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum State {
    Text(String),
    Structured(Structured),
}

// ---------------------------------------------------------------------------
// Answers / response
// ---------------------------------------------------------------------------

/// One typed answer, keyed by the request's qid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Boolean {
        /// P(yes).
        boolean: f64,
    },
    Choice {
        choice: String,
        probabilities: HashMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        legend: HashMap<String, String>,
        probabilities: HashMap<String, f64>,
        confidence: f64,
    },
}

impl Answer {
    /// Confidence for gating decisions. Boolean carries no confidence (a
    /// confident no and a confident yes are equally confident), so it returns
    /// None.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => {
                Some(*confidence)
            }
            Answer::Boolean { .. } => None,
        }
    }

    /// Gate on spread-based confidence: act when `confidence >= threshold`,
    /// else escalate. Boolean has no confidence and therefore always escalates.
    /// The returned value is the acted outcome (winning option key, or the
    /// fractional probability-weighted score); it is None when escalating.
    pub fn decide(&self, threshold: f64) -> (Action, Option<serde_json::Value>) {
        let conf = match self.confidence() {
            Some(c) => c,
            None => return (Action::Escalate, None),
        };
        if conf < threshold {
            return (Action::Escalate, None);
        }
        match self {
            Answer::Choice { choice, .. } => (Action::Act, Some(choice.clone().into())),
            Answer::Score { score, .. } => (Action::Act, Some((*score).into())),
            Answer::Boolean { .. } => (Action::Act, None),
        }
    }
}

/// Outcome of [`Answer::decide`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Act,
    Escalate,
}

/// The response body from `/v1/decisionmaker`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub model: String,
    pub answers: HashMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

// ---------------------------------------------------------------------------
// Health / errors
// ---------------------------------------------------------------------------

/// `GET /health`; 200 only when the model is loaded once (warm).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Health {
    pub status: String,
    pub model: String,
    #[serde(default)]
    pub uptime_seconds: f64,
}

/// Typed error mirroring the spec's `ErrorResponse` shape. `type` values:
/// `validation`, `internal`, `rate_limit`, `overloaded`, `not_ready`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
    #[serde(default)]
    pub details: serde_json::Value,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "decisionmaker {}: {}", self.kind, self.message)
    }
}
impl std::error::Error for ApiError {}

/// Error type for client-side usage violations and transport failures.
#[derive(Debug)]
pub enum Error {
    /// Local invariant violation (e.g. wrong candidate count) before the
    /// request was sent.
    Usage(String),
    /// JSON encode/decode failure.
    Json(serde_json::Error),
    /// HTTP/transport failure (non-contract-level).
    Transport(String),
    /// The server returned a non-2xx with a parseable typed error body.
    Api(ApiError),
    /// The server returned a non-2xx without a parseable error body.
    Http { status: u16, body: String },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Usage(m) => write!(f, "decisionmaker usage: {m}"),
            Error::Json(e) => write!(f, "decisionmaker json: {e}"),
            Error::Transport(m) => write!(f, "decisionmaker transport: {m}"),
            Error::Api(e) => write!(f, "{e}"),
            Error::Http { status, body } => write!(f, "decisionmaker HTTP {status}: {body}"),
        }
    }
}
impl std::error::Error for Error {}

impl Error {
    /// True when the failure is a server-side typed validation error (422).
    pub fn is_validation(&self) -> bool {
        matches!(self, Error::Api(a) if a.kind == "validation")
    }
    /// True when the server is not ready (503 / not_ready).
    pub fn is_not_ready(&self) -> bool {
        matches!(self, Error::Api(a) if a.kind == "not_ready")
    }
}

/// Turns a non-2xx response body into an [`Error`], preferring the typed
/// `ApiError` shape from the spec, else a generic HTTP error.
pub fn decode_error(status: u16, body: &[u8]) -> Error {
    if let Ok(envelope) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(err) = envelope.get("error") {
            if let Ok(api) = serde_json::from_value::<ApiError>(err.clone()) {
                return Error::Api(api);
            }
        }
    }
    Error::Http {
        status,
        body: String::from_utf8_lossy(body).into_owned(),
    }
}

// ---------------------------------------------------------------------------
// Client (async feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "async")]
pub use client::Client;

#[cfg(feature = "async")]
mod client {
    use super::*;

    /// Async client over `reqwest` + `tokio`.
    pub struct Client {
        base_url: String,
        http: reqwest::Client,
    }

    impl Client {
        /// Create a client for a base URL (defaults to the local service).
        pub fn new(base_url: impl Into<String>) -> Self {
            let base_url = base_url.into();
            let base_url = if base_url.is_empty() {
                "http://127.0.0.1:8090".to_string()
            } else {
                base_url.trim_end_matches('/').to_string()
            };
            Self {
                base_url,
                http: reqwest::Client::new(),
            }
        }

        /// Evaluate one `{state, questions}` request; one Answer per qid.
        pub async fn evaluate(&self, req: &Request) -> Result<Response, Error> {
            let resp = self
                .http
                .post(format!("{}/v1/decisionmaker", self.base_url))
                .json(req)
                .send()
                .await
                .map_err(|e| Error::Transport(e.to_string()))?;
            let status = resp.status().as_u16();
            let bytes = resp
                .bytes()
                .await
                .map_err(|e| Error::Transport(e.to_string()))?;
            if status == 200 {
                serde_json::from_slice(&bytes).map_err(|e| {
                    Error::Json(serde_json::Error::io(std::io::Error::other(format!(
                        "decode response (HTTP {status}): {e}"
                    ))))
                })
            } else {
                Err(decode_error(status, &bytes))
            }
        }

        /// Report service readiness.
        pub async fn health(&self) -> Result<Health, Error> {
            let resp = self
                .http
                .get(format!("{}/health", self.base_url))
                .send()
                .await
                .map_err(|e| Error::Transport(e.to_string()))?;
            let status = resp.status().as_u16();
            let bytes = resp
                .bytes()
                .await
                .map_err(|e| Error::Transport(e.to_string()))?;
            if status == 200 {
                serde_json::from_slice(&bytes).map_err(Error::Json)
            } else {
                Err(decode_error(status, &bytes))
            }
        }
    }
}
