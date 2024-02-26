from typing import List, Literal, Optional
from dataclasses import dataclass
import openai
import os
import time
import json

from filelock import FileLock

MODEL_CONFIGS = {   
    "gpt-3.5-turbo-1106": {
        "prompt_cost_per_token": 0.001 / 1000,
        "response_cost_per_token": 0.002 / 1000,
    },
    "gpt-3.5-turbo-0125": {
        "prompt_cost_per_token": 0.0005 / 1000,
        "response_cost_per_token": 0.0015 / 1000,
    },
    "gpt-4-1106-preview": {
        "prompt_cost_per_token": 0.01 / 1000,
        "response_cost_per_token": 0.03 / 1000,
    },
    "gpt-4-0125-preview": {
        "prompt_cost_per_token": 0.01 / 1000,
        "response_cost_per_token": 0.03 / 1000,
    },
}

@dataclass(frozen=True)
class OpenAI:
    model: Literal["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"] = "gpt-3.5-turbo-16k"

    temperature: float = 0.7

    system_prompt: Optional[str] = None

    max_retries = 1
    
    log_file_path = "openai_usage.jsonl"

    def complete(self, conversation: List[str]) -> str:
        config = MODEL_CONFIGS[self.model]
        openai.api_key = os.environ["OPENAI_API_KEY"]
        deployment_name = self.model
        retry_count = 0


        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        for i, prompt in enumerate(conversation):
            messages.append({"role": ("user" if i % 2 == 0 else "assistant"), "content": prompt})

        while True:
            try:
                response = openai.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    temperature=self.temperature,
                )

                break
            except Exception as error:
                if "Please retry after" in str(error):
                    timeout = int(str(error).split("Please retry after ")[1].split(" second")[0]) + 2
                    print(f"Wait {timeout}s before OpenAI API retry ({error})")
                    time.sleep(timeout)
                elif retry_count < self.max_retries:
                    print(f"OpenAI API retry for {retry_count} times ({error})")
                    time.sleep(2)
                    retry_count += 1
                else:
                    print(f"OpenAI API failed for {retry_count} times ({error})")
                    return None

        self.log_usage(config, response.usage)

        generation = response.choices[0].message.content
        return generation

    def log_usage(self, config, usage):
        usage_log = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
        usage_log["prompt_cost"] = config["prompt_cost_per_token"] * usage.prompt_tokens
        usage_log["completion_cost"] = config["response_cost_per_token"] * usage.completion_tokens
        usage_log["cost"] = usage_log["prompt_cost"] + usage_log["completion_cost"]
        usage_log["model"] = self.model
        usage_log["user"] = os.getlogin()

        with FileLock(self.log_file_path + ".lock"):
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(usage_log) + "\n")