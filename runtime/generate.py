from phase2_generate import generate


def run_generation(
    model,
    prompt,
    max_new_tokens,
    temperature,
    top_p,
):

    return generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
