package decisionmaker

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"os/exec"
	"testing"
	"time"
)

// liveReachable reports whether a Decision-Maker service is up at testURL, for
// gating the end-to-end parity test (skips on machines without the server).
func liveReachable(t *testing.T) bool {
	t.Helper()
	c := &http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get(testURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// TestLiveParity asserts the Go client and a plain curl of the SAME payload
// return identical distributions from the live service — the contract-parity
// check. (The golden fixtures in tests/fixtures are ILLUSTRATIVE bodies, not
// the untrained model's real output, so parity is Go-vs-curl, not vs fixture.)
func TestLiveParity(t *testing.T) {
	if !liveReachable(t) {
		t.Skip("no live Decision-Maker service at " + testURL)
	}

	choice, _ := ChoiceQuestion("Which team should handle this?", map[string]any{
		"billing":   "Payments, invoicing, refunds",
		"technical": "Bugs, outages, integrations",
		"sales":     "Pricing, upgrades, new accounts",
	})
	score, _ := ScoreQuestion("How frustrated is the customer?", []any{"Calm", "Frustrated", "Very angry"})

	goResp, err := NewClient(testURL, 0).Evaluate(context.Background(), &Request{
		State: "Help! My payouts have been failing for 3 days.",
		Model: "decisionmaker-latest",
		Questions: map[string]*Question{
			"is_urgent":   BooleanQuestion("Does this convey urgency?", "Explicitly time-sensitive", "No urgency expressed"),
			"department":  choice,
			"frustration": score,
		},
	})
	if err != nil {
		t.Fatalf("live evaluate: %v", err)
	}

	// Same payload via curl, exactly as a shell user would send it.
	curlJSON, err := exec.Command("curl", "-s",
		"-X", "POST", testURL+"/v1/decisionmaker",
		"-H", "Content-Type: application/json",
		"-d", `{"state":"Help! My payouts have been failing for 3 days.","model":"decisionmaker-latest","questions":{"is_urgent":{"type":"boolean","instructions":"Does this convey urgency?","criteria":{"true":"Explicitly time-sensitive","false":"No urgency expressed"}},"department":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments, invoicing, refunds","technical":"Bugs, outages, integrations","sales":"Pricing, upgrades, new accounts"}},"frustration":{"type":"score","instructions":"How frustrated is the customer?","criteria":["Calm","Frustrated","Very angry"]}}}`,
	).CombinedOutput()
	if err != nil {
		t.Fatalf("curl failed: %v", err)
	}
	var curlResp Response
	if err := json.Unmarshal(curlJSON, &curlResp); err != nil {
		t.Fatalf("curl body not a Response: %v\n%s", err, curlJSON)
	}

	if goResp.Model != curlResp.Model {
		t.Errorf("model: Go=%q curl=%q", goResp.Model, curlResp.Model)
	}
	for qid, want := range curlResp.Answers {
		got, ok := goResp.Answers[qid]
		if !ok {
			t.Errorf("Go missing answer %q that curl returned", qid)
			continue
		}
		switch want.Type {
		case "boolean":
			if got.Boolean == nil || math.Abs(*got.Boolean-*want.Boolean) > 1e-9 {
				t.Errorf("%s: boolean Go=%v curl=%v", qid, *got.Boolean, *want.Boolean)
			}
		case "choice":
			if got.Choice != want.Choice {
				t.Errorf("%s: choice Go=%q curl=%q", qid, got.Choice, want.Choice)
			}
			if !mapsEqual(got.Probabilities, want.Probabilities) {
				t.Errorf("%s: probs Go=%v curl=%v", qid, got.Probabilities, want.Probabilities)
			}
		case "score":
			if math.Abs(got.Score-want.Score) > 1e-9 {
				t.Errorf("%s: score Go=%v curl=%v", qid, got.Score, want.Score)
			}
			if !mapsEqual(got.Probabilities, want.Probabilities) {
				t.Errorf("%s: probs Go=%v curl=%v", qid, got.Probabilities, want.Probabilities)
			}
		}
		if got.Confidence != nil {
			if want.Confidence == nil || math.Abs(*got.Confidence-*want.Confidence) > 1e-9 {
				t.Errorf("%s: confidence Go=%v curl=%v", qid, got.Confidence, want.Confidence)
			}
		}
	}

	// Determinism + sums-to-1 on the Go side.
	again, err := NewClient(testURL, 0).Evaluate(context.Background(), &Request{
		State:     "Help! My payouts have been failing for 3 days.",
		Model:     "decisionmaker-latest",
		Questions: map[string]*Question{"department": choice},
	})
	if err != nil {
		t.Fatalf("repeat evaluate: %v", err)
	}
	for qid, a := range goResp.Answers {
		if a.Probabilities == nil {
			continue
		}
		var sum float64
		for _, p := range a.Probabilities {
			sum += p
		}
		if math.Abs(sum-1) > 1e-6 {
			t.Errorf("%s: go probs sum %v != 1", qid, sum)
		}
	}
	if d := again.Answers["department"]; !mapsEqual(d.Probabilities, goResp.Answers["department"].Probabilities) {
		t.Errorf("service non-deterministic: %v vs %v", d.Probabilities, goResp.Answers["department"].Probabilities)
	}
}

// TestLiveMalformed confirms the client surfaces the server's 422 as a typed
// validation error on a live request that violates the contract.
func TestLiveMalformed(t *testing.T) {
	if !liveReachable(t) {
		t.Skip("no live Decision-Maker service at " + testURL)
	}
	c := NewClient(testURL, 0)
	// choice with a single option -> rejected server-side.
	_, err := c.Evaluate(context.Background(), &Request{
		State:     "x",
		Questions: map[string]*Question{"q": {Type: "choice", Criteria: map[string]any{"a": "only"}}},
	})
	if err == nil {
		t.Fatal("expected a validation error")
	}
	ve, ok := err.(*APIError)
	if !ok || ve.Type != ErrTypeValidation {
		t.Fatalf("expected validation *APIError, got %T %v", err, err)
	}
}

func mapsEqual(a, b map[string]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		bv, ok := b[k]
		if !ok || math.Abs(v-bv) > 1e-9 {
			return false
		}
	}
	return true
}
