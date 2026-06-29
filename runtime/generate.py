from phase2_generate import generate


def run_generation(
    model,
    prompt,
    max_new_tokens,
    temperature,
    top_p,
    config=None,
    chat=False,
    system_prompt=None,
    collector=None,
    thinking="off",
):

    return generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        config=config,
        chat=chat,
        system_prompt=system_prompt,
        collector=collector,
    )

