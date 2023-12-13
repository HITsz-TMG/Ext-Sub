import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Union
from dataclasses import dataclass

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from peft import PeftConfig, PeftModel


PROMPT_DICT = {
    "prompt_input": (
        "<|user|>\n"
        "{instruction} {input}\n"
        "<|assistant|>\n"
    ),
    "prompt_no_input": (
        "<|user|>\n"
        "{instruction}\n"
        "<|assistant|>\n"
    ),
}


class GenerationDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data: List[dict]):
        super(GenerationDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, str]:
        return self.data[i]


@dataclass
class DataCollatorForGenerationDataset(object):
    """Collate examples for text generation."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_prompt = [example["text"] for example in batch]
        example_ids = [example["id"] for example in batch]

        tokenizer_results = self.tokenizer(batch_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenizer_results.input_ids
        attention_mask = tokenizer_results.attention_mask

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            example_ids=example_ids
        )


def batch_llm_generate(data_dict_list: List[dict], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, batch_size: int = 1, **kwargs):
    generation_dataset = GenerationDataset(data_dict_list)
    dataloader = DataLoader(
        generation_dataset,
        sampler=SequentialSampler(generation_dataset),
        batch_size=batch_size,
        num_workers=4,
        collate_fn=DataCollatorForGenerationDataset(tokenizer=tokenizer)
    )

    results_list = []
    for batch in tqdm(dataloader, desc="LLM generating"):
        input_ids, attention_mask, example_ids = batch['input_ids'], batch['attention_mask'], batch['example_ids']

        num_return_sequences = kwargs.get("num_return_sequences", 1)

        generated_ids = model.generate(input_ids=input_ids.to(model.device), 
                                       attention_mask=attention_mask.to(model.device),
                                       **kwargs
                                   )
        generated_results = tokenizer.batch_decode(generated_ids[:, input_ids.size(-1):], skip_special_tokens=True)

        generated_results = [generated_results[i:i+num_return_sequences] for i in range(0, len(generated_results), num_return_sequences)]
        results_list.extend([{"id": i, "text": texts} for i, texts in zip(example_ids, generated_results)])

    return results_list


def predict(data: List[dict], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, batch_size: int = 1, generate_num: int = 1):
    data_dict_list = []
    for i, obj in enumerate(tqdm(data)):
        if obj.get("obj") is None or len(obj["input"]) == 0:
            text = PROMPT_DICT["prompt_no_input"].format(instruction=obj["instruction"])
        else:
            text = PROMPT_DICT["prompt_input"].format(instruction=obj["instruction"], input=obj["input"])

        example = {"id": i,
                    "text": text}
        data_dict_list.extend([example] * generate_num)
    
    results_list = batch_llm_generate(data_dict_list, model, tokenizer,
                                      batch_size=batch_size,
                                      max_new_tokens=500,
                                      do_sample=False,
                                      num_beams=1,
                                      num_return_sequences=1,
                                      )
    results_dict = {obj["id"]: [] for obj in results_list}
    for obj in results_list:
        results_dict[obj["id"]].append(obj["text"][0])

    outputs = [results_dict[i] if results_dict.get(i) is not None else [] for i in range(len(data))]

    return outputs


def load_model(model_name_or_path: Union[Path, str]):
    if "lora" in model_name_or_path or "ia3" in model_name_or_path or "prefix" in model_name_or_path:
        config = PeftConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16, device_map="cpu",)
        model = PeftModel.from_pretrained(model, model_name_or_path)
        if "lora" in model_name_or_path:
            model = model.merge_and_unload()
        model = model.cuda()
        
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = model.half()
    model.eval()

    return tokenizer, model


def main(model_name_or_path: Union[Path, str], input_path: Union[Path, str], output_path: Union[Path, str], batch_size: int):
    ## Step 1: Load the tokenizer and model
    tokenizer, model = load_model(model_name_or_path)

    ## Step 2: Load the data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ## Step 3: Use In-Context Learning to predict the labels
    outputs = predict(data, model, tokenizer, batch_size)

    ## Step 4: Store the output results
    for obj, output in zip(data, outputs):
        obj["output"] = [s.strip() for s in output][0]
    
    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="your/path/containing/toxic_test.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="your/output/file/path",
    )
    args = parser.parse_args()
    print(args)

    main(args.model_name_or_path, args.input_path, args.output_path, args.batch_size)
    