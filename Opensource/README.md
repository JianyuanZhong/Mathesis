# Formal Mathematics Prover Evaluation Framework

## Setup Environment

### Requirements
* Supported platform: Linux
* Python 3.10

### Installation

1. **Install Lean 4**  
   Follow the instructions on the [Lean 4 installation page](https://leanprover.github.io/lean4/doc/quickstart.html) to set up Lean 4.

2. **Clone the repository**
```sh
git clone repo
cd repo
# mathlib4
git clone https://github.com/xinhjBrant/mathlib4.git
```

3. **Install dependencies**
```sh
pip install -r requirements.txt
```

4. **Build mathlib4**
```sh
cd mathlib4
lake build
```

5. **Test Lean 4 and mathlib4 installation**
```sh
cd ..
python prover/lean/verifier.py
```
If there is any error, reinstall Lean 4 and rebuild mathlib4.

## Quick Start 

To run inference on our model and reproduce the performance on miniF2F and Gaokao-formal benchmark. Here are some examples using 4 GPUs to evaluate models on the two datasets for Pass@32.

### Input Styles

The framework supports different input styles for various models:

- **default**: Step-by-step solving with explanatory comments (Used by Goedel-Prover-SFT)
- **cot**: CoT-based format with system prompts (Used by Kimina-Prover models)
- **noncot**: Minimalist completion format (Used by DeepSeek-Prover V2 models)

Each style has default generation parameters optimized for that specific input format.

### Example Commands

```sh
# Goedel-Prover-SFT on miniF2F with default style
sh eval/eval.sh -i datasets/minif2f.jsonl -m Goedel-LM/Goedel-Prover-SFT \
  -o results/minif2f/Godel-Prover-SFT -n 32 -g 4 -c 128 -t default

# DeepSeek-Prover on miniF2F with noncot style
sh eval/eval.sh -i datasets/minif2f.jsonl -m deepseek-ai/DeepSeek-Prover-V2-7B \
  -o results/minif2f/DeepSeek-Prover-V2-7B -n 32 -g 4 -c 128 -t noncot

# Kimina-Prover on miniF2F with cot style
sh eval/eval.sh -i datasets/minif2f.jsonl -m AI-MO/Kimina-Prover-Preview-Distill-7B \
  -o results/minif2f/Kimina-Prover-Preview-Distill-7B -n 32 -g 4 -c 128 -t cot

# Goedel-Prover-SFT on Gaokao-formal with default style
sh eval/eval.sh -i datasets/Gaokao-formal.jsonl -m Goedel-LM/Goedel-Prover-SFT \
  -o results/gaokao/Godel-Prover-SFT -n 32 -g 4 -c 128 -t default
```

The results are summarized in `results/[dataset]/[model]/compilation_summarize.json`

### Command Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i` | Dataset path to evaluate | - |
| `-m` | Model name or path | - |
| `-o` | Output directory | - |
| `-t` | Input style: `default`, `cot`, or `noncot` | `default` |
| `-n` | Number of generations per problem (Pass@n) | 32 |
| `-g` | Number of GPUs for inference | 4 |
| `-c` | Number of CPUs for compilation | 128 |
| `-T` | Override default temperature for the selected style | Style-specific |
| `-p` | Override default top-p for the selected style | Style-specific |
| `-M` | Override max tokens for the selected style | Style-specific |

### Default Generation Parameters

| Style | Temperature | Top-p | Max Tokens |
|-------|-------------|-------|------------|
| default | 1.0 | 0.95 | 2048 |
| cot | 0.6 | 0.95 | 16384 |
| noncot | 1.0 | 0.95 | 2048 |

You can override these defaults with the `-T`, `-p`, and `-M` flags.

## Models and Performance

For each model, use the recommended input style for best performance:

| Model | Recommended Input Style |
|-------|-------------------------|
| Goedel-Prover-SFT | default |
| DeepSeek-Prover-V2-7B | noncot |
| Kimina-Prover-Preview-Distill-7B | cot |



