from argparse import ArgumentParser, Namespace, RawTextHelpFormatter

def parse_args() -> Namespace:
    p = ArgumentParser(
        description="Evaluate a language model on a conversational task using multiple APIs",
        formatter_class=RawTextHelpFormatter
    )
    p.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help="The directory of data."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="Where to dump the prediction results."
    )
    p.add_argument(
        "--model_path",
        type=str,
        help="The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it will default to the official HF path.",
    )
    p.add_argument(
        "--model_name",
        type=str,
        choices=["gpt4", "yarn-mistral", "kimi", "claude2", "rwkv", "yi-6b-200k", "yi-34b-200k", "chatglm3"],
        default="gpt4",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",
    )
    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data."
    )
    p.add_argument(
        "--stop_idx",
        type=int,
        help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset."
    )
    p.add_argument("--verbose", action='store_true', help="Enable verbose output")
    p.add_argument("--device", type=str, default="cuda", help="Specify the device to use (e.g., 'cuda' or 'cpu')")
    p.add_argument(
        "--api",
        type=str,
        required=True,
        help="Specify which API to use for evaluation"
    )

    # Add an epilog to provide additional information
    p.epilog = """
Sample usage:
  python eval_multi_api.py --task question_answering --api openai --output_dir ../results --data_dir ../data --verbose

Supported API endpoints:
  - openai
  - anthropic
  - cohere
  - groq
  - openrouter
  - deepseek
  - mistral
  - local_llm
  - llama
  - kobold
  - oobabooga
  - vllm
  - tabbyapi

Make sure to set up your config.txt file with the necessary API keys and configurations.
"""

    return p.parse_args()