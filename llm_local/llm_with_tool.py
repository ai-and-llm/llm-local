import json
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    print(f'get_current_temperature("{location}", "{unit}") <--- CALLED')

    if "Madrid" in location:
        return 22.0

    if "San Francisco" in location:
        return 10.0

    if "Paris" in location:
        return 7.0

    return 2.0


class TemperatureFinderLLM:
    # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            TemperatureFinderLLM.LLM_MODEL_NAME, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            TemperatureFinderLLM.LLM_MODEL_NAME
        )
        self.tools = [get_current_temperature]

    def find_temperature_at(self, location: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location.",
            },
            {
                "role": "user",
                "content": f"Hey, what's the temperature in {location} right now in celsius?",
            },
        ]

        first_response = self.__generate_response_from_model__(messages)

        try:
            data_dict = json.loads(first_response)

            if data_dict["name"] == "get_current_temperature":
                location_param = data_dict["parameters"]["location"]
                unit_param = data_dict["parameters"]["unit"]
                temperature = get_current_temperature(location_param, unit_param)

                tool_call = {
                    "name": "get_current_temperature",
                    "arguments": {"location": location_param, "unit": unit_param},
                }
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [{"type": "function", "function": tool_call}],  # type: ignore
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": "get_current_temperature",
                        "content": str(temperature),
                    }
                )

                second_response = self.__generate_response_from_model__(messages)

                print("Response(with TOOL)")
                return second_response
        except TypeError:
            print("Response (no tool used)")

        return first_response

    def __generate_response_from_model__(self, messages: list[dict[str, Any]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        first_response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )

        return first_response
