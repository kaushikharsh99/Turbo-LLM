from cli.args import parse_args
from config.config import load_config
from runtime.generate import run_generation


def main():

    args = parse_args()
    cfg = load_config(args.config)

    print(
        "\nTurbo-LLM\n"
    )

    model_arg = args.model if args.model is not None else cfg["model"]["path"]
    from loader.model_manager import get_model
    model_path = get_model(model_arg)

    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg["runtime"]["max_new_tokens"]
    temperature = args.temperature if args.temperature is not None else cfg["runtime"]["temperature"]
    top_p = args.top_p if args.top_p is not None else cfg["runtime"]["top_p"]

    if args.collect is None:

        output = run_generation(
            model=model_path,
            prompt=args.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            config=cfg,
            chat=args.chat,
            system_prompt=args.system,
            thinking=args.thinking,
        )

        print("\nOutput:\n")
        print(output)

    else:

        from utils.prompt_loader import PromptLoader
        from utils.data_collector import DataCollector

        prompt_loader = PromptLoader(args.prompts)

        collector = DataCollector(
            output_file=args.output,
        )

        for prompt in prompt_loader.load():

            print("=" * 80)
            print(prompt)
            print("=" * 80)

            run_generation(
                model=model_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                config=cfg,
                chat=args.chat,
                system_prompt=args.system,
                collector=collector,
                thinking=args.thinking,
            )

        if args.collect is not None:
            collector.close()


if __name__=="__main__":

    main()
