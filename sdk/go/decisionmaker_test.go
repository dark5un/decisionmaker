package decisionmaker

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const testURL = "http://127.0.0.1:8090"

// --- constructor validation (no network) ---

func TestConstructors(t *testing.T) {
	q, err := ChoiceQuestion("pick", map[string]any{"a": "one", "b": "two"})
	if err != nil {
		t.Fatalf("valid choice rejected: %v", err)
	}
	if q.Type != "choice" {
		t.Fatalf("choice type = %q", q.Type)
	}

	if _, err = ChoiceQuestion("pick", map[string]any{"a": "only one"}); err == nil {
		t.Fatal("choice with 1 option accepted; want error")
	}
	if _, err = ScoreQuestion("rank", []any{"low"}); err == nil {
		t.Fatal("score with 1 level accepted; want error")
	}
	// 256 options must be rejected.
	big := make(map[string]any, 256)
	for i := 0; i < 256; i++ {
		big[string(rune('a'+i%26))+strings.Repeat("x", i)] = nil
	}
	if _, err = ChoiceQuestion("pick", big); err == nil {
		t.Fatal("choice with 256 options accepted; want error")
	}

	n := BooleanQuestion("yes?", "y desc", "n desc")
	if n.Type != "boolean" {
		t.Fatalf("boolean type = %q", n.Type)
	}
}

func TestRequestMarshal(t *testing.T) {
	q, _ := ChoiceQuestion("pick", map[string]any{"billing": "invoicing", "technical": "bugs"})
	req := &Request{
		State:     "help me",
		Questions: map[string]*Question{"department": q},
	}
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var check map[string]any
	if err := json.Unmarshal(raw, &check); err != nil {
		t.Fatalf("re-unmarshal: %v", err)
	}
	qm := check["questions"].(map[string]any)["department"].(map[string]any)
	if qm["type"] != "choice" {
		t.Fatalf("questions.department.type = %v", qm["type"])
	}
	// model should be omitted when empty (omitempty)
	if _, ok := check["model"]; ok {
		t.Fatal("empty model should be omitted")
	}
}

// --- transport + typed errors against a stub server ---

func TestDecodeError(t *testing.T) {
	e := decodeError(422, []byte(`{"error":{"type":"validation","message":"bad field","details":{"state":"must be non-empty"}}}`))
	api, ok := e.(*APIError)
	if !ok {
		t.Fatalf("expected *APIError, got %T", e)
	}
	if api.Type != ErrTypeValidation || api.Status != 422 {
		t.Fatalf("got %+v", api)
	}
	if api.Details["state"] != "must be non-empty" {
		t.Fatalf("details lost: %v", api.Details)
	}

	// malformed body -> generic HTTP error, not a nil artifact
	e2 := decodeError(500, []byte("not json"))
	api2, ok := e2.(*APIError)
	if !ok || api2.Type != ErrTypeHTTP {
		t.Fatalf("expected http_error fallback, got %+#v (ok=%v)", e2, ok)
	}
}

func TestErrorsIs(t *testing.T) {
	e := decodeError(422, []byte(`{"error":{"type":"validation","message":"nope"}}`))
	if !errors.Is(e, ErrValidation) {
		t.Fatalf("expected errors.Is(validation); got %v", e)
	}
	if errors.Is(e, ErrInternal) {
		t.Fatal("validation should not match internal")
	}
	notReady := decodeError(503, []byte(`{"error":{"type":"not_ready","message":"cold"}}`))
	if !errors.Is(notReady, ErrNotReady) {
		t.Fatalf("expected errors.Is(not_ready); got %v", notReady)
	}
}

func TestEvaluateAgainstStub(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/decisionmaker" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		var req Request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("server decode of incoming request: %v", err)
		}
		// Round-trip: client must send Choice/Score/Boolean under correct qid.
		if _, ok := req.Questions["department"]; !ok {
			t.Fatalf("missing department question; got %#v", req.Questions)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{
			"model":"decisionmaker-latest",
			"answers":{
				"is_urgent":{"type":"boolean","boolean":0.92},
				"department":{"type":"choice","choice":"technical","probabilities":{"billing":0.08,"technical":0.85,"sales":0.07},"confidence":0.82},
				"frustration":{"type":"score","score":1.6,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.05,"1":0.3,"2":0.65},"confidence":0.78}
			},
			"usage":{"input_tokens":312,"output_tokens":48}
		}`))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, 0)
	choice, _ := ChoiceQuestion("which team?", map[string]any{"billing": "inv", "technical": "bugs", "sales": "pricing"})
	resp, err := c.Evaluate(context.Background(), &Request{
		State: "help!",
		Questions: map[string]*Question{
			"department": choice,
		},
	})
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if len(resp.Answers) != 3 {
		t.Fatalf("got %d answers", len(resp.Answers))
	}
	dept := resp.Answers["department"]
	if dept == nil || dept.Type != "choice" || dept.Choice != "technical" {
		t.Fatalf("department answer wrong: %+v", dept)
	}
	urgent := resp.Answers["is_urgent"]
	if urgent.Boolean == nil || *urgent.Boolean != 0.92 {
		t.Fatalf("boolean answer wrong: %+v", urgent)
	}
	if urgent.Confidence != nil {
		t.Fatal("boolean must carry no confidence")
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 312 {
		t.Fatalf("usage wrong: %+v", resp.Usage)
	}
}

func TestMalformedRejectedByStub(t *testing.T) {
	// Client must surface the server's 422 as a validation APIError.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(422)
		_, _ = w.Write([]byte(`{"error":{"type":"validation","message":"invalid","details":{"questions":"at least one"}}}`))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, 0)
	_, err := c.Evaluate(context.Background(), &Request{State: "x", Questions: map[string]*Question{}})
	if err == nil {
		t.Fatal("expected error for empty questions")
	}
	if !errors.Is(err, ErrValidation) {
		t.Fatalf("expected validation error, got %v", err)
	}
}

// --- Decide helper ---

func TestDecide(t *testing.T) {
	hi := 0.9
	lo := 0.3
	act, val := (&Answer{Type: "choice", Choice: "billing", Confidence: &hi}).Decide(0.7)
	if act != ActionAct || val != "billing" {
		t.Fatalf("want act/billing, got %v/%v", act, val)
	}
	act, val = (&Answer{Type: "choice", Choice: "billing", Confidence: &lo}).Decide(0.7)
	if act != ActionEscalate || val != nil {
		t.Fatalf("want escalate, got %v/%v", act, val)
	}
	// boolean never acts (no confidence to gate on)
	act, val = (&Answer{Type: "boolean", Boolean: ptr(0.99)}).Decide(0.7)
	if act != ActionEscalate || val != nil {
		t.Fatalf("boolean must escalate, got %v/%v", act, val)
	}
	// exactly at threshold acts
	at := 0.7
	act, _ = (&Answer{Type: "score", Score: 2.5, Confidence: &at}).Decide(0.7)
	if act != ActionAct {
		t.Fatalf("at-threshold should act, got %v", act)
	}
}

func ptr(f float64) *float64 { return &f }
