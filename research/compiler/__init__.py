"""DecisionMaker Gold Compiler — research/compiler package.

The deterministic, teacher-LLM-free core of the Gold Compiler (plan 05):
rulebook tables -> verified tables -> computed-gold train rows.

Three modules:
  corpus.py    — the decision-table corpus: 10 shortlist tables (re-derived from
                 research/planA/rules.py, the hand-built ground truth) plus
                 evidence-bound held-out tables authored from real SKILL.md files.
  anchors.py   — the ONLY human-in-loop step: hand-marked trigger states per
                 decision (state -> expected outcome) that drive the verifier's
                 behavioral oracle.
  renderers.py — pluggable prose renderers (state -> natural-language phrasings)
                 for the synthesizer.

No teacher LLM ever judges gold. The verifier is a deterministic interpreter +
verbatim span equality over hand-marked anchors.
"""