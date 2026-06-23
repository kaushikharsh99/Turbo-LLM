import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        prog="TurboLLM",
        description="Turbo-LLM Runtime"
    )

    parser.add_argument(
        "--model",
        required=True,
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
        default=50
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95
    )

    parser.add_argument(
        "--benchmark",
        action="store_true"
    )

    return parser.parse_args()
