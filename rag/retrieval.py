import json
import time
from loguru import logger
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import logger, load_dataset
import argparse
from tqdm import tqdm


class Retrieval:
    def __init__(
        self,
        retrieval: str,
        snippets_path: str,
        index_name: str,
        model_path: str,
        npz_root: str = None,
    ):
        self.snippets_path = snippets_path
        self.code_snippets = self.get_code_snippets(self.snippets_path)
        self.index_name = index_name
        self.model_path = model_path
        logger.info(f"Retrieval: {retrieval}")
        logger.info(f"total snippets: {len(self.code_snippets)}")
        logger.info(f"index name: {self.index_name}")

        self.FIM_PRE = "<｜fim▁begin｜>"
        self.FIM_SUF = "<｜fim▁hole｜>"
        self.FIM_MID = "<｜fim▁end｜>"

        if retrieval == "random":
            from rag.random_retrieval import RandomRetrieval

            self.runner = RandomRetrieval(self.code_snippets)

        elif retrieval == "bm25":
            from rag.bm25_retrieval import ESBM25 as BM25Retrieval

            self.runner = BM25Retrieval(self.code_snippets, self.index_name)

        elif retrieval == "jaccard":
            from rag.jaccard_retrieval import JaccardRetrieval

            self.runner = JaccardRetrieval(self.code_snippets, self.model_path)

        elif retrieval in ["gte-large", "gte-base", "gte-small", "unixcoder"]:
            from rag.prompt_retrieval import Retrieval

            self.runner = Retrieval(
                dataset_path=npz_root,
                model_name=model_path,
                max_seq_length=512,
                mode="avg",
                prompt_version="v1",
            )

        elif retrieval.startswith("procc"):
            from rag.prompt_retrieval import Retrieval

            parts = retrieval.split("-")
            embedding_mode, prompt_version = parts[1], parts[2]
            self.runner = Retrieval(
                dataset_path=f"{npz_root}-{prompt_version}-{embedding_mode}",
                model_name=self.model_path,
                max_seq_length=512,
                mode=embedding_mode,
                prompt_version=prompt_version,
            )

        else:
            raise NotImplementedError

    def get_code_snippets(self, snippets_path):
        with open(snippets_path, "r") as f:
            snippets = json.load(f)
        return snippets

    def query_top_k(self, code, best_of=50):
        top_k = self.runner.query_top_k(top_k=best_of, code=code)

        return top_k

    def get_query_code(
        self,
        prefix: str,
        suffix: str,
        groundtruth: str,
        QUERY_LENGTH=60,
        query_type="last_n_lines",
    ) -> str:

        code_context = []
        line_count = 0

        if prefix.strip():
            lines_before = prefix.split("\n")
            start_index = max(0, len(lines_before) - QUERY_LENGTH // 2)
            for line in lines_before[start_index:]:
                code_context.append(line)
                line_count += 1

        if query_type == "groundtruth":
            lines = groundtruth.split("\n")
            for line in lines:
                code_context.append(line)
                line_count += 1

        if suffix.strip():
            lines_after = suffix.split("\n")
            for i, line in enumerate(lines_after):
                if i >= QUERY_LENGTH - line_count:
                    break
                code_context.append(line)

        return "\n".join(code_context)

    def get_query_code_with_fim(
        self,
        prefix: str,
        suffix: str,
        groundtruth: str,
        QUERY_LENGTH=60,
    ) -> str:
        code_before = []
        code_after = []
        line_count = 0

        if prefix.strip():
            lines_before = prefix.split("\n")
            start_index = max(0, len(lines_before) - QUERY_LENGTH // 2)
            for line in lines_before[start_index:]:
                code_before.append(line)
                line_count += 1

        if suffix.strip():
            lines_after = suffix.split("\n")
            for i, line in enumerate(lines_after):
                if i >= QUERY_LENGTH - line_count:
                    break
                code_after.append(line)

        _prefix = "\n".join(code_before) + "\n"
        _suffix = "\n" + "\n".join(code_after)
        return self.FIM_PRE + _prefix + self.FIM_SUF + _suffix + self.FIM_MID


def parse_arguments():
    parser = argparse.ArgumentParser(description="Retrieval Similar Code")
    parser.add_argument("--testset_path", type=str, required=True)
    parser.add_argument("--code_snippets_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--rag_index_name", type=str, required=True, help="vector index name"
    )
    parser.add_argument("--retrieval", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--query_line", type=int, default=10)
    parser.add_argument("--query_type", type=str, default="last_n_lines")
    parser.add_argument("--npz_root", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()

    testsets = load_dataset(args.testset_path)

    retrieval = Retrieval(
        retrieval=args.retrieval,
        snippets_path=args.code_snippets_path,
        index_name=args.rag_index_name,
        model_path=args.model_path,
        npz_root=args.npz_root,
    )

    retrieval_start_time = time.time()
    for sample in tqdm(testsets, desc="Retrieving code snippets"):
        if 'generate' in args.retrieval:
            query_code = retrieval.get_query_code_with_fim(
                prefix=sample["prefix"],
                suffix=sample["suffix"],
                groundtruth=sample["reference"],
                QUERY_LENGTH=args.query_line,
            )
        else:
            query_code = retrieval.get_query_code(
                prefix=sample["prefix"],
                suffix=sample["suffix"],
                groundtruth=sample["reference"],
                QUERY_LENGTH=args.query_line,
                query_type=args.query_type,
            )

        top_k = retrieval.query_top_k(code=query_code, best_of=args.top_k)
        if "rag" not in sample:
            sample["rag"] = {}

        save_rag_name = args.retrieval
        sample["rag"][save_rag_name] = {}
        sample["rag"][save_rag_name]["top_k"] = args.top_k
        sample["rag"][save_rag_name]["query_code"] = query_code
        sample["rag"][save_rag_name]["list"] = top_k
        sample["rag"][save_rag_name]["model_path"] = args.model_path

    retrieval_end_time = time.time()
    logger.info(
        f"Retrieval avg time is {round((retrieval_end_time-retrieval_start_time)/len(testsets), 4)}"
    )

    with open(args.save_path, "w") as f:
        json.dump(testsets, f, indent=4)
    logger.info(f"Saving results to {args.save_path}")


if __name__ == "__main__":
    main()
