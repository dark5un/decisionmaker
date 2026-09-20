"""Starter renderer for the decision "__DECISION__".

The compiler falls back to `_generic` ({state} -> "a is true. b is false.")
when a decision has no registered renderer — that trains the model on TRIVIAL
prose, easy for a text classifier and useless for the "easy-for-human /
hard-for-LLM" goal. Replace the boilerplate below with realistic operator prose:
one natural sentence per field, distractor signals woven in, a closing question.

Loaded automatically by build.py from this directory — no manual RENDER edit
needed. Just keep this filename and the function name unchanged.

Renderer contract: fn(decision_id, st) -> list[str] (>=1 phrasings).
DECISION_ID must match the hyphenated decision_id in tables.jsonl.
"""

DECISION_ID = "__DECISION__"

_FIELDS = ["__FIELDS__"]


def __FUNC__(decision_id, st):
    # Build one realistic sentence per field. Adapt each clause to the field's
    # meaning, and ADD a distractor field's clause so a shallow model is tempted
    # while a human reads the true signal. Two phrasings per state is enough.
    clauses = {}
    for f in _FIELDS:
        v = bool(st.get(f))
        if f == "_replace_me_":                      # example per-field branch
            clauses[f] = ("The filesystem this box writes into is completely "
                          "full." if v else "Disk has ample headroom.")
        else:
            clauses[f] = f"{f} is {v}."              # fallback (trivial)

    core = (" ".join(clauses.values()) +
            " What should be decided for this " + decision_id + " situation?")
    return [core, core + " Decide as the runbook prescribes."]