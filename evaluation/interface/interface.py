from typing import Dict, Union, List, Tuple
from transformers import AutoTokenizer
import copy
import json
from .language import LanguageParser


class Interface:
    def __init__(
        self,
        model_id,
        total_budget=4096,
        max_rag_num=1,
        use_rag=False,
        suffix_first=False,
        model_name="deepseek",
        debug=True,
    ):
        self.total_budget = total_budget
        self.max_rag_num = max_rag_num
        self.suffix_first = suffix_first
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.use_rag = use_rag

        if model_name == "codellama":
            self.FIM_PRE = " <PRE>"
            self.FIM_SUF = " <SUF>"
            self.FIM_MID = " <MID>"
        elif model_name in ["starcoder", "codeqwen"]:
            self.FIM_PRE = "<fim_prefix>"
            self.FIM_SUF = "<fim_suffix>"
            self.FIM_MID = "<fim_middle>"
        elif model_name == "deepseek":
            self.FIM_PRE = "<｜fim▁begin｜>"
            self.FIM_SUF = "<｜fim▁hole｜>"
            self.FIM_MID = "<｜fim▁end｜>"
        elif model_name == "codegemma":
            self.FIM_PRE = "<|fim_prefix|>"
            self.FIM_SUF = "<|fim_suffix|>"
            self.FIM_MID = "<|fim_middle|>"
        else:
            raise NotImplementedError("Model not support yet")

        self.bos_token = ""
        if self.tokenizer.bos_token:
            self.bos_token = self.tokenizer.bos_token

        # 额外的信息总共占用50%的prompt
        self.allocate_budget()
        self.rag_structure = [("rag", -1)] if use_rag else []
        self.prefix_structure = [("before_cursor", -1)]
        self.suffix_structure = [("after_cursor", -1)]

        self.language_parser = LanguageParser

        self.debug = debug

    def allocate_budget(self):
        self.rag_budget = int(self.total_budget * 0.5)
        self.suffix_budget = int(self.total_budget * 0.15)

    def assemble_prompt(
        self,
        prompt_rag: str,
        prompt_prefix: str,
        prompt_suffix: str,
    ) -> str:

        return f"{self.FIM_PRE}{prompt_rag}{prompt_prefix}{self.FIM_SUF}{prompt_suffix}{self.FIM_MID}"

    def gen_prompt(self, context) -> str:

        language = context["language"]
        prompt_rag, remain_budget = self.generate_part_prompt(
            context, self.rag_structure, self.rag_budget, language, self.total_budget
        )
        prompt_prefix, remain_budget = self.generate_part_prompt(
            context,
            self.prefix_structure,
            remain_budget - self.suffix_budget,
            language,
            remain_budget,
        )
        prompt_suffix, remain_budget = self.generate_part_prompt(
            context, self.suffix_structure, remain_budget, language, remain_budget
        )

        # Assemble the final prompt
        prompt = self.assemble_prompt(prompt_rag, prompt_prefix, prompt_suffix)

        # Debug print
        self.debug_print(prompt)

        return prompt

    def debug_print(self, prompt: str):
        if self.debug:
            print(prompt)
            print("-" * 80)
            self.debug = False

    def encode_infilling(self, s: str) -> List[int]:
        if self.model_name == "codellama":
            """Encode a string without an implicit leading space."""
            return self.tokenizer.encode("☺" + s, add_special_tokens=False)[2:]
        else:
            return self.tokenizer.encode(s, add_special_tokens=False)

    def convert_to_commit(self, code: str, comment_symbol: str) -> str:
        return "\n".join([f"{comment_symbol} {line}" for line in code.split("\n")])

    def get_comment_symbols(self, language: str) -> str:
        return self.language_parser.get_comment(language)

    def generate_part_prompt(
        self,
        context: Dict[str, Union[str, List[Dict[str, str]]]],
        structure: List[Tuple[str, int]],
        part_budget: int,
        language: str,
        remain_budget: int,
    ) -> Tuple[str, int]:
        tokenized_context = []
        for key, _ in structure:
            content = context.get(key)

            if content:
                tokenized_content, part_budget = self.tokenize_content(
                    content, language, part_budget, key
                )
                tokenized_context.extend(tokenized_content)

        decoded_context = (
            self.tokenizer.decode(tokenized_context) if tokenized_context else ""
        )
        remain_budget -= len(tokenized_context)
        return decoded_context, remain_budget

    def tokenize_content(
        self,
        content: Union[str, List[Dict[str, str]]],
        language: str,
        part_budget: int,
        key: str,
    ) -> Tuple[List[int], int]:
        if key == "rag" and isinstance(content, list):
            return self.handle_rag_content(content, language, part_budget)

        tokenized_content = (
            self.encode_infilling(content) if isinstance(content, str) else []
        )
        if key == "before_cursor":
            tokenized_content = tokenized_content[-part_budget:]
        else:
            tokenized_content = tokenized_content[:part_budget]

        part_budget -= len(tokenized_content)
        return tokenized_content, part_budget

    def handle_rag_content(
        self, content: List[Dict[str, str]], language: str, part_budget: int
    ) -> Tuple[List[int], int]:
        tokenized_context = []
        comment_symbol = self.get_comment_symbols(language)
        for item in content[: self.max_rag_num]:
            code_with_comments = self.convert_to_commit(item["code"], comment_symbol)
            file_name = "" if item["path"] is None else item["path"].split("/")[-1]
            rag_context = f"{comment_symbol} Compare this snippet from {file_name}: \n{code_with_comments}\n"
            tokenized_content = self.encode_infilling(rag_context)
            if len(tokenized_content) > part_budget:
                break
            tokenized_context.extend(tokenized_content)
            part_budget -= len(tokenized_content)

        return tokenized_context, part_budget
