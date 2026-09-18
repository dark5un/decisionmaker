package decisionmaker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Typed error.type values, mirroring the spec's ErrorResponse enum.
const (
	ErrTypeValidation = "validation" // 422: client broke the contract
	ErrTypeInternal   = "internal"   // 500: engine malfunction
	ErrTypeRateLimit  = "rate_limit" // 429
	ErrTypeOverloaded = "overloaded" // 429 variant
	ErrTypeNotReady   = "not_ready"  // 503: model not loaded yet
	ErrTypeHTTP       = "http_error" // non-contract failure (no parseable body)
)

// APIError is a typed error decoded from the spec's ErrorResponse shape.
// Status is the HTTP status code; Details carries field-level pointers on
// validation failures.
type APIError struct {
	Type    string         `json:"type"`
	Message string         `json:"message"`
	Details map[string]any `json:"details,omitempty"`
	Status  int            `json:"-"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("decisionmaker %s (HTTP %d): %s", e.Type, e.Status, e.Message)
}

// Is reports whether the error chain contains an *APIError of the given type
// (e.g. errors.Is(err, decisionmaker.ErrValidation)).
var (
	ErrValidation = &APIError{Type: ErrTypeValidation, Message: "validation"}
	ErrInternal   = &APIError{Type: ErrTypeInternal, Message: "internal"}
	ErrRateLimit  = &APIError{Type: ErrTypeRateLimit, Message: "rate_limit"}
	ErrNotReady   = &APIError{Type: ErrTypeNotReady, Message: "not_ready"}
)

func (e *APIError) Is(target error) bool {
	t, ok := target.(*APIError)
	return ok && t != nil && t.Type == e.Type
}

// Client talks to a Decision-Maker service over HTTP. It is safe for concurrent
// use; create one and reuse it (the engine is load-once/warm on the server).
type Client struct {
	baseURL string
	http    *http.Client
}

// NewClient returns a client for the given base URL (defaults to
// http://127.0.0.1:8090). timeout <= 0 selects a 30s default.
func NewClient(baseURL string, timeout time.Duration) *Client {
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8090"
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), http: &http.Client{Timeout: timeout}}
}

// Evaluate runs one {state, questions} evaluation and returns one Answer per
// qid. A contract violation surfaces as *APIError with Type "validation".
func (c *Client) Evaluate(ctx context.Context, req *Request) (*Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("decisionmaker: encode request: %w", err)
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/decisionmaker", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("decisionmaker: new request: %w", err)
	}
	hreq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("decisionmaker: post: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("decisionmaker: read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp.StatusCode, raw)
	}
	var out Response
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decisionmaker: decode response (HTTP %d): %w", resp.StatusCode, err)
	}
	return &out, nil
}

// Health reports service readiness. Returns an error (usually *APIError with
// Type "not_ready") unless HTTP 200.
func (c *Client) Health(ctx context.Context) (*Health, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return nil, fmt.Errorf("decisionmaker: new request: %w", err)
	}
	resp, err := c.http.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("decisionmaker: get health: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("decisionmaker: read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp.StatusCode, raw)
	}
	var h Health
	if err := json.Unmarshal(raw, &h); err != nil {
		return nil, fmt.Errorf("decisionmaker: decode health: %w", err)
	}
	return &h, nil
}

// decodeError turns a non-2xx response into an *APIError, falling back to a
// generic HTTP error when the body isn't a parseable ErrorResponse.
func decodeError(status int, raw []byte) error {
	var envelope struct {
		Error *APIError `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Error != nil {
		envelope.Error.Status = status
		if envelope.Error.Message == "" {
			envelope.Error.Message = http.StatusText(status)
		}
		return envelope.Error
	}
	return &APIError{
		Type:    ErrTypeHTTP,
		Message: fmt.Sprintf("%s (no parseable error body)", http.StatusText(status)),
		Status:  status,
	}
}

// fmtErrorCount builds a count-range error for the constructors.
func fmtErrorCount(kind, noun string, got, min, max int) error {
	return fmt.Errorf("decisionmaker: %s criteria must have %d..%d %s (got %d)", kind, min, max, noun, got)
}
