import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        prog="TurboLLM",
        description="Turbo-LLM Runtime"
    )

    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model directory"
    )

    parser.add_argument(
        "--prompt",
        required=True,
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
        "--system",
        type=str,
        default=None,
        help="System prompt (only used with --chat)"
    )

    return parser.parse_args()
