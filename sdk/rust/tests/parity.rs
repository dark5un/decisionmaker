//! Contract-parity tests for the Rust SDK: unit tests (constructors,
//! serialization, decide) plus live tests that must match a curl of the same
//! payload against the real service (skipped when no service is reachable).

use std::collections::HashMap;

use decisionmaker::{
    Action, Answer, ApiError, ChoiceQuestion, Error, Health, BooleanCriteria, Question, Request,
    ScoreQuestion, State,
};

fn test_request() -> Request {
    let mut questions = HashMap::new();
    let mut criteria = HashMap::new();
    criteria.insert(
        "billing".into(),
        Some("Payments, invoicing, refunds".into()),
    );
    criteria.insert(
        "technical".into(),
        Some("Bugs, outages, integrations".into()),
    );
    criteria.insert(
        "sales".into(),
        Some("Pricing, upgrades, new accounts".into()),
    );
    questions.insert(
        "department".to_string(),
        Question::Choice(
            ChoiceQuestion::new(Some("Which team should handle this?".into()), criteria).unwrap(),
        ),
    );
    Request {
        state: State::Text("Help! My payouts have been failing for 3 days.".into()),
        model: Some("decisionmaker-latest".into()),
        questions,
    }
}

fn server_up() -> bool {
    // quick TCP-level probe via std (no reqwest needed for the gate)
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:8090".parse().unwrap(),
        std::time::Duration::from_millis(500),
    )
    .is_ok()
}

// ---------------------------------------------------------------------------
// Serialization shape (offline)
// ---------------------------------------------------------------------------

#[test]
fn question_serializes_with_type_tag() {
    let q = Question::Boolean(decisionmaker::BooleanQuestion {
        instructions: None,
        criteria: Some(BooleanCriteria {
            true_: Some("Explicitly time-sensitive".into()),
            false_: Some("No urgency expressed".into()),
        }),
    });
    let v = serde_json::to_value(q).unwrap();
    let obj = v.as_object().unwrap();
    assert_eq!(obj["type"], "boolean");
    assert_eq!(obj["criteria"]["true"], "Explicitly time-sensitive");
}

#[test]
fn score_question_serializes_ordered() {
    let q = Question::Score(
        ScoreQuestion::new(
            Some("How frustrated?".into()),
            vec!["Calm".into(), "Frustrated".into(), "Very angry".into()],
        )
        .unwrap(),
    );
    let v = serde_json::to_value(q).unwrap();
    let criteria = &v["criteria"];
    assert_eq!(criteria[2], "Very angry");
}

#[test]
fn choice_enforces_count() {
    let one: HashMap<String, Option<decisionmaker::Structured>> = [("a".into(), None)].into();
    assert!(ChoiceQuestion::new(None, one).is_err());
    let two: HashMap<String, Option<decisionmaker::Structured>> =
        [("a".into(), None), ("b".into(), None)].into();
    assert!(ChoiceQuestion::new(None, two).is_ok());
    let too_many: HashMap<String, Option<decisionmaker::Structured>> =
        (0..256).map(|i| (format!("k{i}"), None)).collect();
    assert!(ChoiceQuestion::new(None, too_many).is_err());
}

#[test]
fn score_enforces_level_count() {
    assert!(ScoreQuestion::new(None, vec!["only".into()]).is_err());
    assert!(ScoreQuestion::new(None, vec!["a".into(), "b".into()]).is_ok());
}

#[test]
fn request_serializes_contract_shape() {
    let v = serde_json::to_value(test_request()).unwrap();
    assert_eq!(v["questions"]["department"]["type"], "choice");
    assert_eq!(v["state"], "Help! My payouts have been failing for 3 days.");
}

#[test]
fn answer_decide_gates_on_confidence() {
    let choice = Answer::Choice {
        choice: "billing".into(),
        probabilities: HashMap::new(),
        confidence: 0.9,
    };
    assert_eq!(choice.decide(0.7), (Action::Act, Some("billing".into())));

    let low = Answer::Choice {
        choice: "billing".into(),
        probabilities: HashMap::new(),
        confidence: 0.3,
    };
    assert_eq!(low.decide(0.7), (Action::Escalate, None));

    // boolean has no confidence -> always escalates
    let boolean = Answer::Boolean { boolean: 0.99 };
    assert_eq!(boolean.decide(0.0), (Action::Escalate, None));
    assert!(boolean.confidence().is_none());
}

// ---------------------------------------------------------------------------
// Live parity (Go/curl == Rust); skipped without a reachable service
// ---------------------------------------------------------------------------

#[cfg(feature = "async")]
#[tokio::test]
async fn live_parity_async() {
    if !server_up() {
        eprintln!("SKIP: no live service at 127.0.0.1:8090");
        return;
    }
    let c = decisionmaker::Client::new("");
    let resp = c.evaluate(&test_request()).await.expect("live evaluate");
    assert_eq!(resp.model, "Qwen/Qwen3-0.6B");
    let dept = resp.answers.get("department").unwrap();
    // across the three options
    let probs = match dept {
        Answer::Choice { probabilities, .. } => probabilities,
        other => panic!("expected Choice, got {other:?}"),
    };
    assert_eq!(probs.len(), 3);
    let sum: f64 = probs.values().sum();
    assert!((sum - 1.0).abs() < 1e-6, "probs must sum to 1, got {sum}");
}

#[cfg(feature = "async")]
#[tokio::test]
async fn live_malformed_rejected() {
    if !server_up() {
        eprintln!("SKIP: no live service at 127.0.0.1:8090");
        return;
    }
    let mut questions = HashMap::new();
    // single-option choice -> server rejects with 422 (construct directly,
    // bypassing the local 2..=255 invariant, to exercise the SERVER's check)
    questions.insert(
        "q".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: None,
            criteria: [("a".into(), None)].into(),
        }),
    );
    let c = decisionmaker::Client::new("");
    let err = c
        .evaluate(&Request {
            state: State::Text("x".into()),
            model: None,
            questions,
        })
        .await
        .unwrap_err();
    match &err {
        Error::Api(a) => {
            assert_eq!(a.kind, "validation", "wrong kind: {a:?}");
        }
        other => panic!("expected validation ApiError, got {other:?}"),
    }
}

#[cfg(feature = "blocking")]
#[test]
fn live_parity_blocking() {
    if !server_up() {
        eprintln!("SKIP: no live service at 127.0.0.1:8090");
        return;
    }
    let c = decisionmaker::blocking::Client::new("");
    let resp = c.evaluate(&test_request()).expect("live blocking evaluate");
    let n = resp.answers.len();
    assert_eq!(n, 1);
    let health: Health = c.health().expect("live health");
    assert_eq!(health.status, "ready");
}

#[test]
fn decode_error_prefers_typed() {
    let body = br#"{"error":{"type":"validation","message":"bad","details":{"state":"nope"}}}"#;
    match decisionmaker::decode_error(422, body) {
        Error::Api(a) => assert_eq!(a.kind, "validation"),
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[test]
fn decode_error_falls_back() {
    match decisionmaker::decode_error(500, b"not json") {
        Error::Http { status, .. } => assert_eq!(status, 500),
        other => panic!("expected Http error, got {other:?}"),
    }
}

// helper used implicitly to assert ApiError/serde_json are real types on the bin
#[allow(dead_code)]
fn _types(_a: &ApiError) {}
