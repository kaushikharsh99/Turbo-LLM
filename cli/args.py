import argparse
import os

def parse_args():
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(package_root, "config", "default.yaml")

    parser = argparse.ArgumentParser(
        prog="TurboLLM",
        description="Turbo-LLM Runtime"
    )

    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to config file"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model directory"
    )

    parser.add_argument(
        "--prompt",
        default=None,
        help="Input prompt"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None
    )

    parser.add_argument(
        "--benchmark",
        action="store_true"
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use chat template formatting for instruct models"
    )

    parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default="off",
        help="Qwen thinking mode (default: off)"
    )

    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt (only used with --chat)"
    )

    
    
    parser.add_argument(
        "--collect",
        choices=["routing"],
        default=None,
        help="Collect datasets instead of normal inference."
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to prompts.jsonl"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="routing_dataset.jsonl",
        help="Dataset output file"
    )

    args = parser.parse_args()

    # Validate CLI arguments

    if args.collect is None and args.prompt is None:
        parser.error("--prompt is required unless using --collect.")

    if args.collect is not None and args.prompts is None:
        parser.error("--prompts is required when using --collect.")

    return args