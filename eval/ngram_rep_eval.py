import os
import argparse
import json
from pathlib import Path
from typing import List, Union
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import ngrams
from transformers import AutoTokenizer


def ngram_metrics(token_list: List[int], pad: int = -100):
    if pad in token_list:
        token_list = token_list[:token_list.index(pad)]  # remove possible padding
    stats = defaultdict(float)
    for n in range(1, 5):
        ngs = [ng for ng in ngrams(token_list, n)]
        counter = Counter([ng for ng in ngrams(token_list, n)])
        stats['pct_repeat_%dgrams' % n] = 1.0 - len(counter)/len(ngs) if len(ngs) > 0 else 0
    return stats


def rep_ngram_eval(tokenizer: AutoTokenizer, input_texts: List[str]):
    results = defaultdict(list)
    for text in tqdm(input_texts):
        input_ids = tokenizer(text).input_ids
        stats = ngram_metrics(input_ids)
        for k, v in stats.items():
            results[k].append(v)
    return results


def load_tokenizer(model_name_or_path: Union[Path, str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    return tokenizer


def read_data(input_path: Union[Path, str]):
    if os.path.isdir(input_path):
        # TruthfulQA
        df = pd.read_csv(os.path.join(input_path, "prediction.csv"))
        model_name = os.path.basename(os.path.normpath(input_path))
        # model_name = os.path.splitext(os.path.basename(os.path.normpath(input_path)))[0]
        questions = df["Question"]
        generated_answers = df[model_name]
        data = []
        for q, a in zip(questions, generated_answers):
            data.append({"instruction": q, "input": "", "output": a})
    elif input_path.endswith('.json'):
        # toxic
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError

    return data


def main(input_path: Union[Path, str], output_path: Union[Path, str], model_name_or_path: Union[Path, str]):
    tokenizer = load_tokenizer(model_name_or_path)

    data = read_data(input_path)
    input_texts = [obj["output"] for obj in data]
    
    results = rep_ngram_eval(tokenizer, input_texts)

    model_name = os.path.basename(os.path.normpath(input_path))
    print("#"*3, model_name, np.mean(results["pct_repeat_4grams"]))

    saved_data = {"model": model_name,
                  "pct_repeat_4grams_mean": np.mean(results["pct_repeat_4grams"]),
                  "pct_repeat_3grams_mean": np.mean(results["pct_repeat_3grams"]),
                  "pct_repeat_2grams_mean": np.mean(results["pct_repeat_2grams"]),
                  "pct_repeat_1grams_mean": np.mean(results["pct_repeat_1grams"]),
                  "info": results,
                  }
    
    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saved_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
                        default="your/llama-7b/tokenizer/path/")
    parser.add_argument("--input_path", type=str, 
                        default="the/directory/path/or/file/path/for/generated/results/from/toxic/or/truthfulqa")
    parser.add_argument("--output_path", type=str, 
                        default=None)
    args = parser.parse_args()
    print(args)

    assert args.input_path != args.output_path

    main(args.input_path, args.output_path, args.model_name_or_path)

    torch.cuda.empty_cache()