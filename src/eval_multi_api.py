# eval_multi_api.py
# Description: Evaluate a language model on a conversational task using multiple APIs
#
# Usage: python eval_multi_api.py --task question_answering --api <api_name>> --output_dir ./results --data_dir ./data --verbose
#  API endpoints are defined in the config file (config.txt)
#  The API key for the selected API should be defined in the config file
# APIs Supported are:
#  - openai
#  - anthropic
#  - cohere
#  - groq
#  - openrouter
#  - deepseek
#  - mistral
#  - llamacpp
#  - kobold
#  - oobabooga
#  - vllm
#  - tabbyapi
#
from args import parse_args
import json
from pathlib import Path
import time
from typing import Dict, Any, Optional, List
from eval_utils import (
    create_msgs,
    load_data,
    dump_jsonl,
    iter_jsonl,
    get_answer,
)

# Import API-specific functions
from LLM_API_Calls import (
    chat_with_openai,
    chat_with_anthropic,
    chat_with_cohere,
    chat_with_groq,
    chat_with_openrouter,
    chat_with_deepseek,
    chat_with_mistral
)
from LLM_API_Calls_Local import (
    chat_with_llama,
    chat_with_kobold,
    chat_with_oobabooga,
    chat_with_vllm,
    chat_with_tabbyapi
)

class MultiAPILLMClient:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.api_functions = {
            'openai': chat_with_openai,
            'anthropic': chat_with_anthropic,
            'cohere': chat_with_cohere,
            'groq': chat_with_groq,
            'openrouter': chat_with_openrouter,
            'deepseek': chat_with_deepseek,
            'mistral': chat_with_mistral,
            'llamacpp': chat_with_llama,
            'kobold': chat_with_kobold,
            'oobabooga': chat_with_oobabooga,
            'vllm': chat_with_vllm,
            'tabbyapi': chat_with_tabbyapi
        }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return json.load(f)

    def chat(self, api_name: str, messages: List[Dict[str, str]],
             model: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             **kwargs) -> str:
        if api_name not in self.api_functions:
            raise ValueError(f"Unsupported API: {api_name}")

        api_key = self.config['api_keys'].get(api_name)
        if not api_key:
            raise ValueError(f"API key not found for {api_name}")

        chat_function = self.api_functions[api_name]
        
        # Use config values if not provided in the method call
        model = model or self.config.get('models', {}).get(api_name)
        temperature = temperature or self.config.get('temperature', {}).get(api_name)
        max_tokens = max_tokens or self.config.get('max_tokens', {}).get(api_name)

        # Extract the input_data from messages (assuming it's the last user message)
        input_data = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), "")

        # Prepare common parameters
        common_params = {
            "api_key": api_key,
            "input_data": input_data,
            "custom_prompt_arg": kwargs.get('custom_prompt_arg', ""),
        }

        # Handle specific APIs
        if api_name in ['openai', 'groq', 'openrouter', 'deepseek', 'mistral']:
            return chat_function(**common_params, temp=temperature, system_message=kwargs.get('system_message'))
        elif api_name == 'anthropic':
            return chat_function(**common_params, model=model, max_retries=kwargs.get('max_retries', 3),
                                 retry_delay=kwargs.get('retry_delay', 5), system_prompt=kwargs.get('system_message'))
        elif api_name == 'cohere':
            return chat_function(**common_params, model=model, system_prompt=kwargs.get('system_message'))
        elif api_name == 'llamacpp':
            return chat_function(**common_params, api_url=kwargs.get('api_url'), system_prompt=kwargs.get('system_message'))
        elif api_name == 'kobold':
            return chat_function(**common_params, kobold_api_ip=kwargs.get('kobold_api_ip'),
                                 temp=temperature, system_message=kwargs.get('system_message'))
        elif api_name in ['oobabooga', 'vllm', 'tabbyapi']:
            # These APIs might need special handling, adjust as necessary
            return chat_function(**common_params, **kwargs)
        else:
            # For any other APIs, pass all parameters
            return chat_function(**common_params, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)

def main():
    args = parse_args()
    verbose = args.verbose
    task = args.task
    # New argument for selecting the API
    api_name = args.api

    # Load config from a JSON file
    client = MultiAPILLMClient('config.txt')

    examples = load_data(task)

    result_dir = Path(args.output_dir)
    result_dir.mkdir(exist_ok=True, parents=True)

    output_path = result_dir / f"preds_{task}_{api_name}.jsonl"
    if output_path.exists():
        preds = list(iter_jsonl(output_path))
        start_idx = len(preds)
        stop_idx = len(examples)
    else:
        start_idx = 0
        stop_idx = len(examples)
        preds = []

    start_time = time.time()
    i = start_idx
    while i < stop_idx:
        eg = examples[i]
        msgs, prompt = create_msgs(
            # Use API-specific tokenizer if available
            client.config.get('tokenizer', {}).get(api_name),  
            eg,
            task,
            # Use API-specific model
            model_name=client.config.get('models', {}).get(api_name),
            data_dir=args.data_dir
        )
        if verbose:
            print(f"======== Example {i} =========")
            print("Input text:")
            print(prompt[:300])
            print("...")
            print(prompt[-300:])
            print("==============================")

        # Make prediction
        try:
            response = client.chat(
                api_name, 
                # Pass the full messages list
                msgs,
                custom_prompt_arg=prompt,
                temperature=client.config.get('temperature', {}).get(api_name),
                max_tokens=client.config.get('max_tokens', {}).get(api_name),
                system_message=client.config.get('system_messages', {}).get(api_name)
            )
            preds.append(
                {
                    "id": i,
                    "prediction": response,
                    "ground_truth": get_answer(eg, task),
                }
            )
            # Save result
            dump_jsonl(preds, output_path)
            print("Time spent:", round(time.time() - start_time))
            print(response)
            time.sleep(20)
            i += 1
        except Exception as e:
            print("ERROR:", e)
            print("Retrying...")
            time.sleep(60)

if __name__ == "__main__":
    main()
