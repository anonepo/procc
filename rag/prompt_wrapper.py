from loguru import logger


class PromptWrapper:
    def __init__(
        self,
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        prompt_version: str = "v1",
    ):
        self.prompt_version = prompt_version
        
        # pure embedding
        if self.prompt_version == "v1":
            self.prompt_prefix = ""
            self.prompt_suffix = ""
        # hypy line.
        elif self.prompt_version == "v2":
            self.prompt_prefix = "This code snippets of "
            self.prompt_suffix = " means"

        # embedding, for context,
        elif self.prompt_version == "v3":
            self.prompt_prefix = "Embedding the following code snippets:\n"
            self.prompt_suffix = ""
        
        # summariztion, generation.
        elif self.prompt_version == "v4":
            self.prompt_prefix = "Summarize the following code snippets:\n"
            self.prompt_suffix = ""


        logger.info(f"Prompt prefix{repr(self.prompt_prefix)}")
        logger.info(f"Prompt suffix{repr(self.prompt_suffix)}")

    def wrapper(self, code: str) -> str:
        prompt = f"{self.prompt_prefix}{code}{self.prompt_suffix}"
        return prompt

    def de_wrapper(self, prompt: str) -> str:
        assert prompt.startswith(self.prompt_prefix) and prompt.endswith(
            self.prompt_suffix
        ), print(prompt)

        if len(self.prompt_suffix) == 0:
            # If yes, return the string without slicing from the end
            return prompt[len(self.prompt_prefix) :]
        else:
            # If no, slice the string as originally intended
            return prompt[len(self.prompt_prefix) : -len(self.prompt_suffix)]

    def apply_prompt(self, example):
        example["code"] = self.wrapper(example["code"])
        return example

    def prompt(self) -> str:
        return f"{self.prompt_prefix}[code]{self.prompt_suffix}"
