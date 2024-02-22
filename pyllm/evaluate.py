import argparse
import logging

from typing import List, Dict
from tabulate import tabulate


from pyllm.utils.registry import CLIENT_REGISTRY, METHOD_REGISTRY, DATASET_REGISTRY
from pyllm.interfaces import CodeGenerator
from pyllm.function_datasets.base import FunctionDataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate(
    methods: List[str], datasets: List[str], client_name: str, client_args: dict
):
    client = CLIENT_REGISTRY.build(client_name, **client_args)
    methods: Dict[str, CodeGenerator] = {
        method: METHOD_REGISTRY.build(method, client=client) for method in methods
    }
    datasets: Dict[str, FunctionDataset] = {
        dataset: DATASET_REGISTRY.build(dataset) for dataset in datasets
    }

    results = {}

    for method_name, method in methods.items():
        # logging.info(f"Evaluating method: {method_name}")
        results[method_name] = {}
        for dataset_name, dataset in datasets.items():
            # logging.info(f"Evaluating dataset: {dataset_name}")
            correct, total = 0, 0
            pbar = tqdm(dataset, desc=f"{dataset_name} - {method_name}")
            for row in pbar:
                try:
                    method.def_function(
                        row.prompt, unit_tests=row.unit_tests, use_cached=False
                    )
                    correct += 1
                except Exception as e:
                    logging.error(
                        f"Error evaluating method {method_name} on dataset {dataset_name}: {e}"
                    )
                finally:
                    total += 1
                    pbar.set_postfix({"Accuracy": correct / total})

            results[method_name][dataset_name] = correct / total
            # logging.info(f"Method {method_name} Accuracy: {correct / total:.2%}")

    # Print summary
    print("\nEvaluation Summary:")
    print(
        tabulate(
            [[k] + list(v.values()) for k, v in results.items()],
            headers=list(datasets.keys()),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", "-m", required=True, type=str)
    parser.add_argument("--datasets", "-d", required=True, type=str)
    parser.add_argument("--client", "-c", default="openai", type=str)
    parser.add_argument("--client-args", "--client_args", "-ca", default="")

    args = parser.parse_args()

    if args.methods == "all":
        methods = list(METHOD_REGISTRY._classes_dict.keys())
    else:
        methods = args.methods.split(",")

    if args.datasets == "all":
        datasets = list(DATASET_REGISTRY._classes_dict.keys())
    else:
        datasets = args.datasets.split(",")
    client_name = args.client

    client_args = args.client_args.split(",")
    client_args = {
        arg.split("=")[0]: "=".join(arg.split("=")[1:]) for arg in client_args
    }

    print(f"Evaluating methods: {methods}")
    print(f"Evaluating datasets: {datasets}")

    evaluate(
        methods=methods,
        datasets=datasets,
        client_name=client_name,
        client_args=client_args,
    )
