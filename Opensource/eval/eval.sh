#!/bin/bash

INPUT_PATH=datasets/minif2f.jsonl
MODEL_PATH=Goedel-LM/Goedel-Prover-SFT
OUTPUT_DIR=results/minif2f/Godel-Prover-SFT
N=32
CPU=128
GPU=4
FIELD=complete
INPUT_STYLE=default  # Options: default, cot, noncot
TEMPERATURE=""  # Will use default for the selected style
TOP_P=""        # Will use default for the selected style
MAX_TOKENS=""   # Will use default for the selected style

while getopts ":i:m:o:s:n:c:g:t:T:p:M:" opt; do
  case $opt in
    i) INPUT_PATH="$OPTARG"
    ;;
    m) MODEL_PATH="$OPTARG"
    ;;
    o) OUTPUT_DIR="$OPTARG"
    ;;
    n) N="$OPTARG"
    ;;
    c) CPU="$OPTARG"
    ;;
    g) GPU="$OPTARG"
    ;;
    t) INPUT_STYLE="$OPTARG"
    ;;
    T) TEMPERATURE="$OPTARG"
    ;;
    p) TOP_P="$OPTARG"
    ;;
    M) MAX_TOKENS="$OPTARG"
    ;;
  esac
done

# Create temperature flag if provided
TEMP_FLAG=""
if [ ! -z "$TEMPERATURE" ]; then
  TEMP_FLAG="--temperature $TEMPERATURE"
fi

# Create top_p flag if provided
TOP_P_FLAG=""
if [ ! -z "$TOP_P" ]; then
  TOP_P_FLAG="--top_p $TOP_P"
fi

# Create max_tokens flag if provided
MAX_TOKENS_FLAG=""
if [ ! -z "$MAX_TOKENS" ]; then
  MAX_TOKENS_FLAG="--max_tokens $MAX_TOKENS"
fi

echo "Running step 1: Model inference with $INPUT_STYLE style"
python -m eval.step1_inference \
  --input_path ${INPUT_PATH} \
  --model_path ${MODEL_PATH} \
  --output_dir $OUTPUT_DIR \
  --n $N \
  --gpu $GPU \
  --input_style $INPUT_STYLE \
  $TEMP_FLAG $TOP_P_FLAG $MAX_TOKENS_FLAG

echo "Running step 2: Code compilation"
INPUT_FILE=${OUTPUT_DIR}/to_inference_codes.json
COMPILE_OUTPUT_PATH=${OUTPUT_DIR}/code_compilation.json
python -m eval.step2_compile \
  --input_path $INPUT_FILE \
  --output_path $COMPILE_OUTPUT_PATH \
  --cpu $CPU

echo "Running step 3: Summarizing results"
SUMMARIZE_OUTPUT_PATH=${OUTPUT_DIR}/compilation_summarize.json
python -m eval.step3_summarize_compile \
  --input_path $COMPILE_OUTPUT_PATH \
  --output_path $SUMMARIZE_OUTPUT_PATH \
  --field ${FIELD}

echo "Evaluation completed! Results saved to $SUMMARIZE_OUTPUT_PATH"