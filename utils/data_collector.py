import json
import os


class DataCollector:
    """
    Collects routing information for one generated token and
    generates one training sample per layer.

    Output:
        JSON Lines (.jsonl)
        One line = one training sample.
    """

    def __init__(
        self,
        output_file="routing_dataset.jsonl",
        num_layers=None,
        buffer_size=5000,
    ):
        self.output_file = output_file
        self.num_layers = num_layers
        self.buffer_size = buffer_size

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        self.fp = open(output_file, "a", encoding="utf-8")

        self.buffer = []
        self.current_token = None



    def begin_token(
        self,
        token_id,
        token_text,
        position,
        generation_step,
        thinking,
        temperature,
        top_p,
    ):

        if self.num_layers is None:
            raise RuntimeError(
                "Collector num_layers has not been initialized."
            )

        self.current_token = {
            "token_id": token_id,
            "token_text": token_text,

            # Absolute transformer position
            "position": position,

            "thinking": thinking,
            # Decode step (1,2,3,...)
            "generation_step": generation_step,
            "temperature": temperature,
            "top_p": top_p,

            "routes": [None] * self.num_layers
        }

    def record_layer(
        self,
        layer_id,
        experts,
        scores,
    ):

        self.current_token["routes"][layer_id] = {
            "experts": list(experts),
            "scores": list(scores),
        }


    def finish_token(self):

        routes = self.current_token["routes"]

        assert all(route is not None for route in routes), (
            "Not all layer routes were recorded."
        )

        for layer in range(self.num_layers):

            current = routes[layer]

            sample = {

                "token_id": self.current_token["token_id"],

                "token_text": self.current_token["token_text"],

                "position": self.current_token["position"],
                "generation_step": self.current_token["generation_step"],
                "thinking": self.current_token["thinking"],

                "temperature": self.current_token["temperature"],
                "top_p": self.current_token["top_p"],

                "layer_id": layer,

                "current_route": {
                    "experts": current["experts"],
                    "scores": current["scores"],
                },

                "past_layer_routes": [

                    {
                        "experts": route["experts"],
                        "scores": route["scores"],
                    }

                    for route in routes[:layer]
                ],

                "future_layer_routes": [

                    {
                        "experts": route["experts"],
                        "scores": route["scores"],
                    }

                    for route in routes[layer + 1:]
                ],
            }

            self.buffer.append(sample)

        self.current_token = None

        if len(self.buffer) >= self.buffer_size:
            self.flush()


    def flush(self):

        for sample in self.buffer:
            self.fp.write(json.dumps(sample))
            self.fp.write("\n")

        self.fp.flush()
        self.buffer.clear()


    def close(self):

        if self.buffer:
            self.flush()

        self.fp.close()