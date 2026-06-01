# Instruction Following

IFBench instruction following evaluation (300 problems).

**Setup:**
```bash
# Set your IFBench path (used throughout this guide)
export IFBENCH_PATH=~/IFBench

# 1. Clone IFBench
mkdir -p $(dirname $IFBENCH_PATH)
git clone https://github.com/allenai/IFBench.git $IFBENCH_PATH

# 2. Install dependencies
pip install absl-py spacy syllapy emoji unicodedata2 nltk
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# 3. Speed up: Comment out the slow download line in instructions.py
sed -i "s/^download('en_core_web_sm')$/# download('en_core_web_sm')/" $IFBENCH_PATH/instructions.py

# 4. Verify installation
python -c "import sys; sys.path.insert(0, '$IFBENCH_PATH'); import instructions_registry; print(f'IFBench loaded: {len(instructions_registry.INSTRUCTION_DICT)} instructions')"
```

**Run:**
```bash
export IFBENCH_PATH=~/IFBench
llm-in-sandbox benchmark --task instruct_follow
llm-in-sandbox benchmark --task instruct_follow --mode llm   # vanilla LLM
```

**Metric:** IFBench loose pass rate.

**Dataset:** `daixuancheng/llm-in-sandbox-bench` (config: `instruct_follow`)
