# ESI-Bench Recipe Splits

`recipe_train10.txt` and `recipe_validation10.txt` are deterministic, disjoint
development manifests supplied so the contrib recipe runs without untracked
files. Each contains one question from every official top-level category. Both
are disjoint from `reported_eval231.txt`, the evaluation manifest associated
with the author-reported result snapshot.

These files are not claimed to be the exact 10/10 manifests used to produce the
paper tables because those original manifests were not recoverable. Set
`ESI_TRAIN_SPLIT` and `ESI_VALIDATION_SPLIT` to explicit replacements when
reproducing another run. `recipe_metadata.json` records the selected task
families and pinned upstream commit without copying answers or simulator
metadata.
