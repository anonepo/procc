import json
import sys
from datasets import load_from_disk
from tqdm import tqdm
import argparse
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")


def tokenize_nltk(text):
    words = word_tokenize(text)
    return words


def split_sample_with_fim(content, window_sizes, slice_sizes=1, model_type="deepseek"):
    snippets = content.split("\n")
    snippet_list = []

    for idx in range(1, len(snippets) - 1, slice_sizes):
        prefix_lines = snippets[:idx]
        current_line = snippets[idx]
        suffix_lines = snippets[idx + 1 :]

        if len(current_line.strip()) < 3:
            continue

        # 当前行不应该是注释：
        if (
            current_line.strip().startswith("//")
            or current_line.strip().startswith("/*")
            or current_line.strip().startswith("*")
        ):
            continue

        line_count = 0
        prefix_context = []
        suffix_context = []

        start_index = max(0, len(prefix_lines) - window_sizes // 2)
        for line in prefix_lines[start_index:]:
            prefix_context.append(line)
            line_count += 1

        for i, line in enumerate(suffix_lines):
            if i >= window_sizes - line_count:
                break
            suffix_context.append(line)

        block_lines = prefix_context + [current_line] + suffix_context

        _prefix = "\n".join(prefix_context) + "\n"
        _suffix = "\n" + "\n".join(suffix_context)

        if model_type == "deepseek":
            combined_snippet = (
                "<｜fim▁begin｜>"
                + _prefix
                + "<｜fim▁hole｜>"
                + _suffix
                + "<｜fim▁end｜>"
            )
        else:
            raise NotImplementedError

        snippet_list.append(
            {
                "code": combined_snippet,
                "start_line": idx + 1,
                "type": "snippet_with_hypo",
                "block": "\n".join(block_lines),
            }
        )

    return snippet_list


def split_sample(content, window_sizes=60, slice_sizes=60):
    snippets = content.split("\n")
    snippet_list = []
    for l in range(0, len(snippets), slice_sizes):
        lines = snippets[l : l + window_sizes]
        snippet = "\n".join(lines)
        tokenized_snippet = tokenize_nltk(snippet)
        if len(tokenized_snippet) > 0:
            snippet_list.append(
                {
                    "code": snippet,
                    "start_line": l + 1,
                    "type": "snippets",
                    "block": snippet,
                }
            )
    return snippet_list


def process_dataset(args):
    ds = load_from_disk(args.data_dir)[args.mode]

    samples = []
    for content, language, path in zip(ds["code"], ds["language"], ds["path"]):
        sample = {"code": content, "language": language, "path": path}
        samples.append(sample)

    print(f"Processing {len(samples)} code files...")

    all_snippets = []
    for sample in tqdm(samples, total=len(samples), desc="Construct code corpus"):
        if args.hypo:
            snippets = split_sample_with_fim(
                sample["code"], args.window_sizes, args.slice_sizes, args.model_type
            )
        else:
            snippets = split_sample(sample["code"], args.window_sizes, args.slice_sizes)
        for snippet in snippets:
            all_snippets.append(
                {
                    "code": snippet["code"],
                    "language": sample["language"],
                    "path": sample["path"],
                    "start_line": snippet["start_line"],
                    "block": snippet["block"],
                }
            )

    print(f"totol {len(all_snippets)} snippets")

    with open(args.output_file, "w") as f:
        json.dump(all_snippets, f, indent=4)

    print(f"data save to {args.output_file}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--window_sizes", type=int, default=60)
    parser.add_argument("--slice_sizes", type=int, default=1)
    parser.add_argument("--hypo", type=bool, default=False)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--model_type", default="deepseek")

    return parser.parse_args()


def main():
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
