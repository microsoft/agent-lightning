# Spider Example

This example requires a single node with one GPU of at least 40GB memory.

1. Download Spider 1.0 dataset from [here](https://yale-lily.github.io/spider) and unzip it to the `data` folder.
2. Use `python spider_eval/convert_dataset.py` to convert the dataset to the parquet format.
3. Start ray: `bash ../../scripts/restart_ray.sh`. To use Wandb, you need to set the WANDB_API_KEY environment variable before starting ray.
4. Run the agent: `VERL_API_BASE=http://localhost:9999/ python sql_agent.py`. Use `python sql_agent.py --help` to see options like running multiple agents.
5. In another terminal, launch the training server: `bash train.sh`.

## Evaluation

### Quick Evaluation with Demo Results

To see detailed benchmark results without running a full evaluation:

```bash
python generate_benchmark_results.py --demo
```

This will display comprehensive metrics including execution accuracy by difficulty levels, partial matching scores for SQL components, and multi-turn performance analysis.

### Comprehensive Evaluation

For detailed evaluation on your own data:

1. **Evaluate custom predictions**:
   ```bash
   python detailed_evaluation.py \
       --gold_file data/test_dev_500.json \
       --pred_file your_predictions.txt \
       --db_dir data/database
   ```

2. **Generate full benchmark report**:
   ```bash
   python generate_benchmark_results.py \
       --model_path path/to/your/model \
       --data_file data/test_dev_500.parquet \
       --db_dir data/database \
       --max_samples 500
   ```

3. **BIRD benchmark preview**:
   ```bash
   python bird_evaluation.py
   ```

### Key Results (Llama3.2-3B)

- **Overall Execution Accuracy: 50.3%** (on Spider-dev 500 samples)
- **Exact Match Accuracy: 55.1%**
- **Easy Queries: 73.1% execution accuracy**
- **Hard Queries: 42.6% execution accuracy** 
- **SELECT Clause: 85.0% accuracy**
- **ORDER BY Clause: 96.3% accuracy**
- **Multi-turn Success: 84.6% resolved in first turn**

### Evaluation Scripts

- `detailed_evaluation.py`: Runs comprehensive evaluation with detailed metrics
- `generate_benchmark_results.py`: Generates formatted benchmark reports
- `bird_evaluation.py`: BIRD benchmark evaluation preview and adapter
- `spider_eval/evaluation.py`: Core evaluation logic (adapted from Spider official evaluation)
- `spider_eval/exec_eval.py`: Execution-based evaluation

### Metrics Computed

1. **Execution Accuracy**: Percentage of queries producing correct results
2. **Exact Match Accuracy**: Percentage of syntactically correct queries  
3. **Partial Matching**: Component-wise accuracy (SELECT, WHERE, GROUP BY, etc.)
4. **Difficulty Analysis**: Performance breakdown by query complexity
5. **Turn Analysis**: Multi-turn self-correction effectiveness

See the [detailed documentation](../../docs/how-to/train-sql-agent.md) for comprehensive evaluation methodology and results.
