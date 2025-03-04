import json
from json import JSONDecodeError

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
    print(f"get_current_temperature(\"{location}\", \"{unit}\") <--- CALLED")
    return 22.3

class TemperatureFinderLLM:

    def __init__(self):
        pass

    def find_temperature_at(self, location: str) -> str:
        llm_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tools = [get_current_temperature]

        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16)

        messages = [
            {"role": "system",
             "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
            {"role": "user", "content": f"Hey, what's the temperature in {location} right now in celsius?"}
        ]

        inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")
        inputs = {k: v for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=128)
        first_response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        try:
            data_dict = json.loads(first_response)

            if data_dict["name"] == "get_current_temperature":
                location_param = data_dict["parameters"]["location"]
                unit_param = data_dict["parameters"]["unit"]
                temperature = get_current_temperature(location_param, unit_param)

                tool_call = {"name": "get_current_temperature",
                             "arguments": {"location": "Paris, France", "unit": "celsius"}}
                messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
                messages.append({"role": "tool", "name": "get_current_temperature", "content": str(temperature)})

                inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                       return_dict=True,
                                                       return_tensors="pt")
                inputs = {k: v for k, v in inputs.items()}
                out = model.generate(**inputs, max_new_tokens=128)
                second_response = tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

                print("Response(with tool))")
                return second_response
        except JSONDecodeError or TypeError:
            print("Response (no tool used)")
            return first_response