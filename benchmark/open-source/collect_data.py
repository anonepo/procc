import os
import argparse
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import shutil


def split_dataset(dataset, test_ratio):
    train_test_split = dataset.train_test_split(test_size=test_ratio)
    return train_test_split


def collect_files(path, extensions):
    collected_files = []
    for root, _, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            # 检查文件是否为符号链接
            if not os.path.islink(full_path):
                ext = file.rsplit(".", 1)[-1] if "." in file else ""
                if ext.lower() in extensions:
                    collected_files.append(full_path)
    return collected_files


def create_dataset(file_paths, languages_map):
    data = []
    for file_path in tqdm(file_paths, desc="Creating datasets"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                code = file.read()
                extension = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
                language = languages_map.get(extension)
                if language:  # Only include files with recognized extensions
                    repo = file_path.split(os.sep)[0]
                    sample = {
                        "path": file_path,
                        "repository": repo,
                        "language": language,
                        "code": code,
                    }
                    data.append(sample)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return Dataset.from_list(data)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Collect code files and create a dataset."
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the directory containing code files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the dataset",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to model",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    return parser.parse_args()


def split_dataset(dataset):
    train_test_val_split = dataset.train_test_split(
        test_size=0.2
    )  # 80% train, 20% for test+val
    test_val_split = train_test_val_split["test"].train_test_split(
        test_size=0.5
    )  # Split the 20% into 10% test and 10% val
    return DatasetDict(
        {
            "train": train_test_val_split["train"],
            "test": test_val_split["test"],
            "val": test_val_split["train"],
        }
    )


def main():
    args = parse_arguments()
    extensions = ["java"]
    languages_map = {
        "java": "Java",
    }

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)  # This removes the directory and all its contents
        print(f"Removed existing data at {args.save_path}")

    file_paths = collect_files(args.path, extensions)
    dataset = create_dataset(file_paths, languages_map)

    dataset_splits = split_dataset(dataset)
    dataset_splits.save_to_disk(args.save_path)

    print(f"Dataset saved to {args.save_path}")
    print(f"Train samples: {len(dataset_splits['train'])}")
    print(f"Test samples: {len(dataset_splits['test'])}")
    print(f"Val samples: {len(dataset_splits['val'])}")


if __name__ == "__main__":
    main()
