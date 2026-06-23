from cli.args import parse_args
from runtime.generate import run_generation


def main():

    args = parse_args()

    print(
        "\nTurbo-LLM\n"
    )

    output = run_generation(
        model=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(
        "\nOutput:\n"
    )

    print(output)


if __name__=="__main__":

    main()
