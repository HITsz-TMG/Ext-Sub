import os
import argparse
import copy
import logging
import json
import jsonlines
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
import transformers
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

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


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict: List[dict], tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

def calculate_cross_entropy(lm_logits: torch.Tensor, labels: torch.LongTensor):
    labels = labels.to(lm_logits.device)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction="sum")    # using sum due to the various 

    loss_list = []
    for shift_logit, shift_label in zip(shift_logits, shift_labels):
        loss = loss_fct(shift_logit, shift_label)
        loss_list.append(loss.detach().cpu())
    return loss_list


def calculate_loss_list(model: AutoModelForCausalLM, dataloader: DataLoader):
    loss_list = []

    for batch in tqdm(dataloader, desc="computing"):
        with torch.no_grad():
            input_ids, labels, attention_mask = batch['input_ids'], batch['labels'], batch['attention_mask']
            outputs = model(input_ids=input_ids.to(model.device),
                            attention_mask=attention_mask.to(model.device))
            
            lm_logits = outputs.logits
            neg_log_likelihood = calculate_cross_entropy(lm_logits, labels)

            loss_list.extend(neg_log_likelihood)

    loss_list = [loss.item() for loss in loss_list]
    return loss_list


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm(strings, desc="tokenizing")
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_models(model_name_or_path: Union[Path, str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    # tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.model_max_length = 2048
    
    if "lora" in model_name_or_path or "ia3" in model_name_or_path or "prefix" in model_name_or_path:
        config = PeftConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16, device_map="cpu",)
        model = PeftModel.from_pretrained(model, model_name_or_path)
        if "lora" in model_name_or_path:
            model = model.merge_and_unload()
        model = model.cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    return tokenizer, model


def read_data(input_path: Union[Path, str]):
    qa_data_path = os.path.join(input_path, "qa_data.json")
    summarization_data_path = os.path.join(input_path, "summarization_data.json")
    dialogue_data_path = os.path.join(input_path, "dialogue_data.json")

    with jsonlines.open(qa_data_path, "r") as f:
        raw_qa_data = [obj for obj in f]
    with jsonlines.open(summarization_data_path, "r") as f:
        raw_summarization_data = [obj for obj in f]
    with jsonlines.open(dialogue_data_path, "r") as f:
        raw_dialogue_data = [obj for obj in f]
    
    qa_true_data = []
    qa_false_data = []
    for i, obj in enumerate(raw_qa_data):
        qa_true_data.append({"instruction": obj["question"],
                             "input": "",
                             "output": obj["right_answer"]})
        qa_false_data.append({"instruction": obj["question"],
                              "input": "",
                              "output": obj["hallucinated_answer"]})
    
    dialogue_true_data = []
    dialogue_false_data = []
    for i, obj in enumerate(raw_dialogue_data):
        dialogue_true_data.append({"instruction": obj["dialogue_history"],
                             "input": "",
                             "output": obj["right_response"]})
        dialogue_false_data.append({"instruction": obj["dialogue_history"],
                              "input": "",
                              "output": obj["hallucinated_response"]})
    
    summarization_true_data = []
    summarization_false_data = []
    for i, obj in enumerate(raw_summarization_data):
        summarization_true_data.append({"instruction": "Summarize the following document:\n" + obj["document"],
                             "input": "",
                             "output": obj["right_summary"]})
        summarization_false_data.append({"instruction": "Summarize the following document:\n" + obj["document"],
                              "input": "",
                              "output": obj["hallucinated_summary"]})
    
    return qa_true_data, qa_false_data, summarization_true_data, summarization_false_data, dialogue_true_data, dialogue_false_data


def get_data_score(data: List[dict], tokenizer: AutoTokenizer, model: AutoModelForCausalLM, batch_size: int):
    dataset = SupervisedDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        num_workers=4,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    )

    scores = calculate_loss_list(model, dataloader)
    return scores


def compute_precision(loss_list: List[float]):
    true_loss, false_loss = loss_list[:len(loss_list)//2], loss_list[len(loss_list)//2:]
    assert len(true_loss) == len(false_loss)

    precision = []
    for t, f in zip(true_loss, false_loss):
        precision.append(int(t < f))
    
    return np.mean(precision)


def main(input_path: Union[Path, str], output_path: Union[Path, str], model_name_or_path: Union[Path, str], batch_size: int = 4):
    qa_true_data, qa_false_data, summarization_true_data, summarization_false_data, dialogue_true_data, dialogue_false_data = read_data(input_path)
    
    tokenizer, model = load_models(model_name_or_path)

    summarization_loss = get_data_score(summarization_true_data + summarization_false_data, tokenizer, model, batch_size)
    qa_loss = get_data_score(qa_true_data + qa_false_data, tokenizer, model, batch_size)
    
    qa_precision = compute_precision(qa_loss)
    summarization_precision = compute_precision(summarization_loss)

    model_name = os.path.splitext(os.path.basename(os.path.normpath(input_path)))[0]
    print(model_name, qa_precision, summarization_precision)

    saved_data = {"qa_precision": qa_precision,
                  "summarization_precision": summarization_precision,
                  }
    
    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saved_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
                        default="")
    parser.add_argument("--batch_size", type=int, 
                        default=1)
    parser.add_argument("--input_path", type=str, 
                        default="your/directory/containing/HaluEval/data")
    parser.add_argument("--output_path", type=str, 
                        default="your/output/file/path")
    args = parser.parse_args()
    print(args)

    main(args.input_path, args.output_path, args.model_name_or_path, args.batch_size)

    torch.cuda.empty_cache()