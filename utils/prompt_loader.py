import json


class PromptLoader:

    def __init__(self, filename):
        self.filename = filename

    def load(self):

        with open(self.filename, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                obj = json.loads(line)

                yield obj["prompt"]