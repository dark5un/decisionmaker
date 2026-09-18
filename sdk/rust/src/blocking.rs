//! Blocking client over `reqwest::blocking`. Enabled with the `blocking`
//! feature. Same types as the async client; only the transport differs.

use super::{Error, Health, Request, Response};

/// Blocking client over `reqwest::blocking::Client`.
pub struct Client {
    base_url: String,
    http: reqwest::blocking::Client,
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
            http: reqwest::blocking::Client::new(),
        }
    }

    /// Evaluate one `{state, questions}` request; one Answer per qid.
    pub fn evaluate(&self, req: &Request) -> Result<Response, Error> {
        let resp = self
            .http
            .post(format!("{}/v1/decisionmaker", self.base_url))
            .json(req)
            .send()
            .map_err(|e| Error::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let bytes = resp.bytes().map_err(|e| Error::Transport(e.to_string()))?;
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
    pub fn health(&self) -> Result<Health, Error> {
        let resp = self
            .http
            .get(format!("{}/health", self.base_url))
            .send()
            .map_err(|e| Error::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let bytes = resp.bytes().map_err(|e| Error::Transport(e.to_string()))?;
        if status == 200 {
            serde_json::from_slice(&bytes).map_err(Error::Json)
        } else {
            Err(decode_error(status, &bytes))
        }
    }
}

use super::decode_error;
