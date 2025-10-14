# Train the First Agent with Agent-Lightning

## Example Results

We run the example with the following hyper-parameters:

* `val_batch_size` = 10
* `gradient_batch_size` = 4
* `beam_width` = 2
* `branch_factor` = 2
* `beam_rounds` = 2

The model used in agents is `gpt-4.1-nano`, and the model used to critique the prompts and rewrite the prompts are `gpt-5-mini` and `gpt-4.1-mini` respectively.

The validation accuracy on the 29 samples of datasets steadily increase from 0.569 (baseline) to 0.638 (after round 1) to 0.721 (after round 2). (In another run, it's from 0.534 to 0.628 to 0.645). The tuning takes around 10 minutes with 8 runners.
