package decisionmaker

// Action is the outcome of gating a decision on an answer's confidence.
// The plan's contract: use confidence as a GATE (act / review / escalate),
// never as an absolute correctness probability.
type Action string

const (
	ActionAct      Action = "act"
	ActionEscalate Action = "escalate"
)

// Decide gates on the answer's spread-based confidence. If confidence >=
// threshold the decision acts and returns the acted value; otherwise it
// escalates for human review. Boolean has no confidence, so it always escalates
// (a confident no and a confident yes are equally confident — there is no
// peakedness to gate on).
//
// Returned acted value by type:
//
//	choice -> the winning option key (string)
//	score  -> the fractional probability-weighted level (float64)
//	boolean   -> not reached (always escalates)
//
// The value is nil when the action is escalate.
func (a *Answer) Decide(threshold float64) (Action, any) {
	if a == nil || a.Confidence == nil {
		return ActionEscalate, nil
	}
	if *a.Confidence < threshold {
		return ActionEscalate, nil
	}
	switch a.Type {
	case "choice":
		return ActionAct, a.Choice
	case "score":
		return ActionAct, a.Score
	default:
		return ActionAct, nil
	}
}
