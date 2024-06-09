import asyncio
import json
import os
import sys
import time
import traceback
import pandas as pd
import torch
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.metrics import metrics
from tqdm import tqdm
from interface.interface import Interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def formatter(record):
    format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n"

    return format

logger.remove()  
logger.add(sys.stdout, format=formatter)  

def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument(
        "--model_name",
        choices=["codellama", "starcoder", "deepseek", "codegemma", "codeqwen"],
        default="codellama",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--testset_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--use_rag", type=bool, default=False)
    parser.add_argument("--retrieval", type=str, default="bm25")
    parser.add_argument("--total_budget", type=int, default=4096)
    parser.add_argument("--max_rag_num", type=int, default=1)
    parser.add_argument("--use_vllm", type=bool, default=False)
    parser.add_argument("--group_key", type=str, default="language")
    return parser.parse_args()


def run_eval_pipeline(args: ArgumentParser) -> int:

    model_path = Path(args.model_path)
    if not model_path.exists() or not model_path.is_dir():
        logger.error(f"Invalid model {model_path}")
        return -1

    try:
        testsets = json.load(open(args.testset_path, "r"))
        testsets = testsets[: args.sample_num] if args.sample_num > 0 else testsets
        logger.info(f"Loaded testset with {len(testsets)} cases")

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if args.model_name == "starcoder":
            stop_sequences = ["<|endoftext|>", "<file_sep>"]
        elif args.model_name == "codellama":
            stop_sequences = [
                " <EOT>",
                "<EOT>",
                tokenizer.eos_token,
                tokenizer.eot_token,
            ]
        elif args.model_name == "deepseek":
            stop_sequences = ["<｜end▁of▁sentence｜>", tokenizer.eos_token]
        elif args.model_name == "codegemma":
            stop_sequences = ['<|file_separator|>', tokenizer.eos_token]
        elif args.model_name == "codeqwen":
            stop_sequences = [tokenizer.eos_token, "<file_sep>"]

        context_data = []
        for testset in testsets:
            retrievals = testset.get("rag", {}).get("list", [])
            context = {
                "before_cursor": testset["prefix"],
                "after_cursor": testset["suffix"],
                'language': testset['language'],
                'path': testset['file_path'],
                "rag": retrievals,
            }
            context_data.append(context)

        interface = Interface(
            model_id=model_path,
            total_budget=args.total_budget-20,
            max_rag_num=args.max_rag_num,
            use_rag=args.use_rag,
            suffix_first=False,
            model_name=args.model_name,
            debug=True,
        )

        logger.info(f"Generating {args.model_name} inference prompt...")
        inputs = []
        for context in context_data:
            inputs.append(interface.gen_prompt(context))

        if args.use_vllm:
            from vllm import LLM, SamplingParams
            llm = LLM(
                model=args.model_path,
                tensor_parallel_size=args.gpus,
                max_model_len=args.total_budget,
                # max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )

            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                stop=stop_sequences,
            )

            start_time = time.time()
            results = llm.generate(inputs, sampling_params)
            results = [[output.outputs[0].text] for output in results]
            avg_time = round((time.time() - start_time) / len(inputs) if inputs else 0, 2)


    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return -1

    df_data = []
    metric_columns = []
    for idx, (testset, output) in enumerate(zip(testsets, results)):
        output = output[0]
        reference: str = testset["reference"]

        eval_metrics = metrics(output, reference, tokenizer)

        if metric_columns == []:
            metric_columns = list(eval_metrics.keys())

        data_entry = {
            "task_id": testset.get("task_id", 0),
            "language": testset.get("language", "java"),
            "file_path": testset.get("file_path", "file_path"),
            "line_ind": testset.get("line_ind", 0),
            "type": testset.get("type", "default"),
            "reference": reference,
            "output": output,
            "repo": testset.get("repo", ""),
        }

        data_entry.update(eval_metrics)
        df_data.append(data_entry)

    df = pd.DataFrame(df_data)
    group_key = args.group_key
    logger.info(f"Group by key: {group_key}")

    metric_aggregations = {metric: "mean" for metric in metric_columns}
    average_metrics = df.groupby([group_key]).agg(metric_aggregations).round(4)
    counts = df.groupby([group_key]).size()
    average_metrics = average_metrics.assign(count=counts).reset_index()
    average_metrics = average_metrics.sort_values(
        by=["count", group_key], ascending=[False, True]
    )
    overall_avg = {group_key: "all"}

    for metric in metric_columns:
        overall_avg[metric] = round(df[metric].mean(), 4)

    overall_avg["count"] = df.shape[0]
    average_metrics = pd.concat(
        [average_metrics, pd.DataFrame([overall_avg])], ignore_index=True
    )

    print(tabulate(average_metrics, headers="keys", tablefmt="pretty"))

    log_res = {"avg_time": avg_time, "testset_size": len(testsets)}
    logger.info(log_res)
    logger.info(f"Saving results to {args.save_path}")

    save_data = df.to_dict(orient="records")
    if args.save_path:
        json.dump(save_data, open(args.save_path, "w"), indent=4, ensure_ascii=True)
    return 0


def main():
    args = parse_args()
    ret = run_eval_pipeline(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
