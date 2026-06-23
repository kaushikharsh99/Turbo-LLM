from pathlib import Path
from huggingface_hub import snapshot_download


ROOT = (
    Path.home()
    /
    ".turbollm"
)

MODELS = (
    ROOT
    /
    "models"
)

MODELS.mkdir(
    parents=True,
    exist_ok=True
)


def get_model(
    model_id
):

    if Path(model_id).exists():
        model_path = str(Path(model_id).resolve())
        print(f"\nModel:\n{model_id}")
        print(f"\nLocation:\n{model_path}\n")
        return model_path

    local = (
        MODELS
        /
        model_id.replace(
            "/",
            "_"
        )
    )

    print(f"\nModel:\n{model_id}")
    print(f"\nLocation:\n{local}\n")

    if local.exists():

        print(
            "Using cached model"
        )

        return str(
            local.resolve()
        )

    print(
        "Downloading..."
    )

    snapshot_download(
        repo_id=model_id,
        local_dir=str(
            local
        )
    )

    return str(
        local.resolve()
    )
