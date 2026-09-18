// Package decisionmaker is a thin typed client for the Decision-Maker service
// (/v1/decisionmaker). It mirrors the wire contract in spec/openapi.yaml:
// Choice / Boolean / Score questions in, per-candidate probability
// distributions out, with spread-based confidence and typed errors.
package decisionmaker

// Question is a typed question. `Type` is the discriminant; `Criteria`
// carries the type-specific body (see BooleanQuestion/ChoiceQuestion/
// ScoreQuestion constructors). `Instructions` is optional on all three.
type Question struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions,omitempty"`
	Criteria     any    `json:"criteria,omitempty"`
}

// BooleanQuestion returns a yes/no question. criterionTrue / criterionFalse
// are optional {true,false} descriptions of what a yes/no means.
func BooleanQuestion(instructions any, criterionTrue, criterionFalse any) *Question {
	criteria := map[string]any{}
	if criterionTrue != nil {
		criteria["true"] = criterionTrue
	}
	if criterionFalse != nil {
		criteria["false"] = criterionFalse
	}
	return &Question{Type: "boolean", Instructions: instructions, Criteria: criteria}
}

// ChoiceQuestion returns an option question; criteria must have 2..255
// option-key -> description (or nil) entries. Order is preserved by the
// server and passed to the model unchanged.
func ChoiceQuestion(instructions any, criteria map[string]any) (*Question, error) {
	if len(criteria) < 2 || len(criteria) > 255 {
		return nil, fmtErrorCount("choice", "options", len(criteria), 2, 255)
	}
	return &Question{Type: "choice", Instructions: instructions, Criteria: criteria}, nil
}

// ScoreQuestion returns an ordinal question; criteria is an ordered array of
// level descriptions, LOW to HIGH, with 2..10 levels. The ordering must be
// inferable from the descriptions only.
func ScoreQuestion(instructions any, criteria []any) (*Question, error) {
	if len(criteria) < 2 || len(criteria) > 10 {
		return nil, fmtErrorCount("score", "levels", len(criteria), 2, 10)
	}
	return &Question{Type: "score", Instructions: instructions, Criteria: criteria}, nil
}

// Request is a call to /v1/decisionmaker. State may be a string, object, or
// array. Questions maps qid -> question; qids are never sent to the model
// and answers come back under the same keys.
type Request struct {
	State     any                  `json:"state"`
	Model     string               `json:"model,omitempty"`
	Questions map[string]*Question `json:"questions"`
}

// Response is the result of an Evaluate call.
type Response struct {
	Model   string             `json:"model"`
	Answers map[string]*Answer `json:"answers"`
	Usage   *Usage             `json:"usage"`
}

// Usage reports token counts for the evaluation.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Answer is one typed answer, keyed by the request's qid.
// Only the fields for its Type are populated: Boolean sets Boolean only;
// Choice sets Choice + Probabilities + Confidence;
// Score sets Score + Legend + Probabilities + Confidence.
// Boolean carries NO confidence (a confident no and a confident yes are
// equally confident).
type Answer struct {
	Type          string             `json:"type"`
	Boolean          *float64           `json:"boolean,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         float64            `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

// Health is the response from GET /health; 200 only when the model is warm.
type Health struct {
	Status        string  `json:"status"`
	Model         string  `json:"model"`
	UptimeSeconds float64 `json:"uptime_seconds"`
}
