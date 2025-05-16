import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='Path to input JSONL file')
parser.add_argument('--model_path', type=str, help='Path to model')
parser.add_argument('--output_dir', type=str, help='Output directory')
parser.add_argument('--n', default=32, type=int, help='Number of generations per input')
parser.add_argument('--gpu', default=1, type=int, help='Number of GPUs per instance')
parser.add_argument('--input_style', default="default", choices=["default", "cot", "noncot"], 
                    help='Input style: default (step-by-step), cot (with system prompt), or noncot (just complete)')
parser.add_argument('--temperature', type=float, help='Override the default temperature for the selected style')
parser.add_argument('--top_p', type=float, help='Override the default top_p for the selected style')
parser.add_argument('--max_tokens', type=int, help='Override the default max_tokens for the selected style')

args = parser.parse_args()


def load_data(data_path):
    """Load data from JSONL file"""
    data_list = []
    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)
    return data_list

def prepare_inputs(data_list, input_style, tokenizer=None):
    """Prepare model inputs based on the specified style"""
    model_inputs = []
    
    for data in data_list:
        formal_statement = data['formal_statement']
        
        if input_style == "noncot":
            # noncot style - just complete the code
            prompt = "Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}".format(
                formal_statement=formal_statement.replace(' sorry', '\n'),
            )
        elif input_style == "cot":
            # CoT style with system prompt
            input_prompt = "Think about and solve the following problem step by step in Lean 4.\n# Formal statement:\n```lean4\n{formal_statement}\n```\n".format(
                formal_statement=formal_statement,
            )
            messages = [
                {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
                {"role": "user", "content": input_prompt}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Default style - step by step with explanatory comments
            prompt = "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{formal_statement}".format(
                formal_statement=formal_statement.replace(' sorry', '\n'),
            )
        
        model_inputs.append(prompt)
    
    return model_inputs

def extract_code(inputs):
    """Extract code from model output"""
    try:
        return re.search(r'```lean4\n(.*?)\n```', inputs, re.DOTALL).group(1)
    except:
        return "None"

def main():
    # Load data
    data_list = load_data(args.input_path)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Prepare inputs based on the specified style
    model_inputs = prepare_inputs(data_list, args.input_style, tokenizer)
    
    # Set default parameters based on input style
    style_config = {
        "cot": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 16384,
        },
        "default": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 2048,
        },
        "noncot": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 2048,
        }
    }
    
    # Use style-specific parameters but allow command-line overrides
    temp = args.temperature if args.temperature is not None else style_config[args.input_style]["temperature"]
    top_p = args.top_p if args.top_p is not None else style_config[args.input_style]["top_p"]
    max_tokens = args.max_tokens if args.max_tokens is not None else style_config[args.input_style]["max_tokens"]
    
    print(f"Using generation parameters - Temperature: {temp}, Top-p: {top_p}, Max tokens: {max_tokens}")
    
    # Set model parameters based on style
    if args.input_style == "cot":
        model = LLM(
            model=args.model_path, 
            seed=65,
            trust_remote_code=True, 
            swap_space=8, 
            tensor_parallel_size=args.gpu,
            gpu_memory_utilization=0.95,
        )
    else:
        model = LLM(
            model=args.model_path, 
            seed=65, 
            trust_remote_code=True, 
            swap_space=8, 
            tensor_parallel_size=args.gpu, 
            gpu_memory_utilization=0.95,
        )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        n=args.n,
    )
    
    # Generate outputs
    model_outputs = model.generate(
        model_inputs,
        sampling_params,
        use_tqdm=True,
    )
    
    assert len(model_outputs) == len(model_inputs)
    
    # Process outputs
    to_inference_codes = []
    for i in range(len(data_list)):
        data_list[i]["model_input"] = model_inputs[i]
        data_list[i]["model_outputs"] = [output.text for output in model_outputs[i].outputs]
        
        # Extract code based on input style
        if args.input_style == "cot":
            # For cot style, check if the </think> tag is present
            full_codes = []
            for output in model_outputs[i].outputs:
                full_text = model_inputs[i] + output.text
                if '</think>' in full_text:
                    # If '</think>' is present, split and extract code from the second part
                    text_parts = full_text.split('</think>', 1)
                    code = extract_code(text_parts[1])
                else:
                    code = "None"
                full_codes.append(code)
            data_list[i]["full_code"] = full_codes
        else:
            # For other styles, extract code from the full text
            data_list[i]["full_code"] = [extract_code(model_inputs[i] + output.text) for output in model_outputs[i].outputs]
        
        # Prepare for inference
        problem_id = data_list[i].get("problem_id", data_list[i].get("name"))
        to_inference_codes += [{"name": problem_id, "code": code} for code in data_list[i]["full_code"]]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save full records
    output_file_path = f'{args.output_dir}/full_records.json'
    print(f"Outputting to {output_file_path}")
    with open(output_file_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)
    
    # Save codes for inference
    toinfer_file_path = f'{args.output_dir}/to_inference_codes.json'
    print(f"Outputting to {toinfer_file_path}")
    with open(toinfer_file_path, 'w') as json_file:
        json.dump(to_inference_codes, json_file, indent=4)

if __name__ == "__main__":
    main()