import os
import shutil
import argparse
from pathlib import Path
from typing import Union, List, Tuple

import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig, load_peft_weights
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING


def copy_folder(src_folder: Union[Path, str], dst_folder: Union[Path, str], except_names: List[str]=None):
    assert src_folder != dst_folder

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    files = os.listdir(src_folder)
    for file_name in tqdm(files, desc="Copy directory"):
        if except_names is not None and file_name in except_names:
            continue
        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)
        if os.path.isdir(src_file):
            copy_folder(src_file, dst_file)
        else:
            shutil.copy2(src_file, dst_file)


def lora2full_matrix(input_path: Union[Path, str], output_path: Union[Path, str]):
    # Step 1: processing adapter weights
    input_adapter_path = os.path.join(input_path, "adapter_model.bin")
    output_adapter_path = os.path.join(output_path, "adapter_model.bin")

    adapter_weights = torch.load(input_adapter_path)

    r = []
    for param_key, param_value in tqdm(adapter_weights.items(), desc="Converting"):
        if "lora_B" in param_key:
            param_key_down = param_key.replace("lora_B", "lora_A")
            param_key_up= param_key

            data_type = adapter_weights[param_key_down].dtype

            full_matrix = torch.matmul(adapter_weights[param_key_up].to(torch.float32), 
                                       adapter_weights[param_key_down].to(torch.float32))
            assert full_matrix.size(0) == full_matrix.size(1)
            adapter_weights[param_key_down] = torch.eye(full_matrix.size(0), 
                                                      device=full_matrix.device, 
                                                      dtype=data_type)
            adapter_weights[param_key_up] = full_matrix.to(data_type)

            r.append(full_matrix.size(0))
        
    assert all(x == r[0] for x in r)
    
    torch.save(adapter_weights, output_adapter_path)

    # Step 2: processing config
    config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig.from_pretrained(input_path).peft_type
            ].from_pretrained(input_path)
    scaling = config.lora_alpha / config.r
    config.r = r[0]
    config.lora_alpha = scaling * config.r

    config.save_pretrained(output_path)


def merge_lora_weight(input_adapter_path: Union[Path, str]) -> Tuple[dict, int]:
    """
    Convert lora Down and Up matrix into a merged matrix and an Identity Matrix
    """
    adapter_weights = load_peft_weights(input_adapter_path)

    r_list = []
    for param_key in tqdm(adapter_weights.keys(), desc="Merging"):
        if "lora_B" in param_key:
            param_key_A = param_key.replace("lora_B", "lora_A")
            param_key_B = param_key

            data_type = adapter_weights[param_key_A].dtype

            full_matrix = torch.matmul(adapter_weights[param_key_B].to(torch.float32), 
                                       adapter_weights[param_key_A].to(torch.float32))
            # assert full_matrix.size(0) == full_matrix.size(1)
            adapter_weights[param_key_A] = torch.eye(full_matrix.size(1), 
                                                      device=full_matrix.device, 
                                                      dtype=data_type)
            adapter_weights[param_key_B] = full_matrix.to(data_type)

            r_list.append(full_matrix.size(0))
        
    assert all(x == r_list[0] for x in r_list)
    return adapter_weights, r_list[0]


def weight_subtraction(input_path_1: Union[Path, str], input_path_2: Union[Path, str], alpha: float, output_path: Union[Path, str]):
    peft_type = PeftConfig.from_pretrained(input_path_1).peft_type

    if peft_type == "LORA":
        adapter_weights_1, r_1 = merge_lora_weight(input_path_1)
        adapter_weights_2, r_2 = merge_lora_weight(input_path_2)
        assert r_1 == r_2

        for param_key in tqdm(adapter_weights_1.keys(), desc="Subtraction"):
            if "lora_B" in param_key:
                adapter_weights_1[param_key] = adapter_weights_1[param_key] - alpha * adapter_weights_2[param_key]

        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))

        # Config processing: r, lora_alpha
        config = PEFT_TYPE_TO_CONFIG_MAPPING[
                    PeftConfig.from_pretrained(input_path_1).peft_type
                ].from_pretrained(input_path_1)
        scaling = config.lora_alpha / config.r
        config.r = r_1
        config.lora_alpha = scaling * config.r

        config.save_pretrained(output_path)
    elif peft_type == "IA3":
        adapter_weights_1 = torch.load(os.path.join(input_path_1, "adapter_model.bin"))
        adapter_weights_2 = torch.load(os.path.join(input_path_2, "adapter_model.bin"))

        for param_key in tqdm(adapter_weights_1.keys(), desc="Subtraction"):
            adapter_weights_1[param_key] = adapter_weights_1[param_key] + alpha * (2.0 * torch.ones_like(adapter_weights_2[param_key]) - adapter_weights_2[param_key])
        
        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))
    elif peft_type == "PREFIX_TUNING":
        adapter_weights_1 = torch.load(os.path.join(input_path_1, "adapter_model.bin"))
        adapter_weights_2 = torch.load(os.path.join(input_path_2, "adapter_model.bin"))

        for param_key in tqdm(adapter_weights_1.keys(), desc="Subtraction"):
            adapter_weights_1[param_key] = adapter_weights_1[param_key] - alpha * adapter_weights_2[param_key]
        
        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))
    else:
        raise NotImplementedError(peft_type)
    

def weigth_addition(input_path_1: Union[Path, str], input_path_2: Union[Path, str], alpha: float, output_path: Union[Path, str]):
    peft_type = PeftConfig.from_pretrained(input_path_1).peft_type

    if peft_type == "LORA":
        adapter_weights_1, r_1 = merge_lora_weight(input_path_1)
        adapter_weights_2, r_2 = merge_lora_weight(input_path_2)
        assert r_1 == r_2

        for param_key in tqdm(adapter_weights_1.keys(), desc="Addition"):
            if "lora_B" in param_key:
                adapter_weights_1[param_key] = adapter_weights_1[param_key] + alpha * adapter_weights_2[param_key]

        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))

        # Config processing: r, lora_alpha
        config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type].from_pretrained(input_path_1)
        scaling = config.lora_alpha / config.r
        config.r = r_1
        config.lora_alpha = scaling * config.r

        config.save_pretrained(output_path)
    elif peft_type == "IA3":
        adapter_weights_1 = torch.load(os.path.join(input_path_1, "adapter_model.bin"))
        adapter_weights_2 = torch.load(os.path.join(input_path_2, "adapter_model.bin"))

        for param_key in tqdm(adapter_weights_1.keys(), desc="Addition"):
            # adapter_weights_1[param_key] = adapter_weights_1[param_key] + alpha * (adapter_weights_2[param_key] - torch.ones_like(adapter_weights_2[param_key]))
            adapter_weights_1[param_key] = adapter_weights_1[param_key] + adapter_weights_2[param_key]
        
        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))
    elif peft_type == "PREFIX_TUNING":
        adapter_weights_1 = torch.load(os.path.join(input_path_1, "adapter_model.bin"))
        adapter_weights_2 = torch.load(os.path.join(input_path_2, "adapter_model.bin"))

        for param_key in tqdm(adapter_weights_1.keys(), desc="Subtraction"):
            adapter_weights_1[param_key] = adapter_weights_1[param_key] + alpha * adapter_weights_2[param_key]
        
        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))
    else:
        raise NotImplementedError(peft_type)


def weigth_extraction_before_subtraction(input_path_1: Union[Path, str], input_path_2: Union[Path, str], alpha: float, output_path: Union[Path, str]):
    peft_type = PeftConfig.from_pretrained(input_path_1).peft_type

    if peft_type == "LORA":
        adapter_weights_1, r_1 = merge_lora_weight(input_path_1)
        adapter_weights_2, r_2 = merge_lora_weight(input_path_2)
        assert r_1 == r_2

        for param_key in tqdm(adapter_weights_1.keys(), desc="Projection"):
            if "lora_B" in param_key:
                weight_1 = adapter_weights_1[param_key]
                weight_2 = adapter_weights_2[param_key]
                norm_weight_1 = weight_1 / torch.norm(weight_1, dim=-1, keepdim=True)
                norm_weight_2 = weight_2 / torch.norm(weight_2, dim=-1, keepdim=True)

                common_feature = norm_weight_1 + norm_weight_2
                norm_common_feature = common_feature / torch.norm(common_feature, dim=-1, keepdim=True)

                project_weight = []
                for i in range(weight_2.size(0)):
                    project_weight.append(torch.dot(weight_2[i], norm_common_feature[i]) * norm_common_feature[i])
                project_weight = torch.stack(project_weight)
                
                adapter_weights_1[param_key] = adapter_weights_1[param_key] - alpha * adapter_weights_2[param_key] + alpha * project_weight

        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))

        # Config processing: r, lora_alpha
        config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type].from_pretrained(input_path_1)
        scaling = config.lora_alpha / config.r
        config.r = r_1
        config.lora_alpha = scaling * config.r

        config.save_pretrained(output_path)

    elif peft_type == "IA3":
        adapter_weights_1 = torch.load(os.path.join(input_path_1, "adapter_model.bin"))
        adapter_weights_2 = torch.load(os.path.join(input_path_2, "adapter_model.bin"))
        
        for param_key in tqdm(adapter_weights_1.keys(), desc="Projection"):
            squeeze_size = adapter_weights_1[param_key].shape.index(1)

            delta_weights_1 = (adapter_weights_1[param_key] - torch.ones_like(adapter_weights_1[param_key])).squeeze(squeeze_size)
            delta_weights_2 = (adapter_weights_2[param_key] - torch.ones_like(adapter_weights_2[param_key])).squeeze(squeeze_size)
            norm_weight_1 = delta_weights_1 / torch.norm(delta_weights_1, dim=-1, keepdim=True)
            norm_weight_2 = delta_weights_2 / torch.norm(delta_weights_2, dim=-1, keepdim=True)

            common_feature = norm_weight_1 + norm_weight_2
            norm_common_feature = common_feature / torch.norm(common_feature, dim=-1, keepdim=True)
            
            project_weight = torch.dot(delta_weights_2, norm_common_feature) * norm_common_feature

            merged_delta_weight = (delta_weights_1 - delta_weights_2 + alpha * project_weight).unsqueeze(squeeze_size)

            adapter_weights_1[param_key] = merged_delta_weight + torch.ones_like(merged_delta_weight)
        
        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))
    elif peft_type == "PREFIX_TUNING":
        adapter_weights_1 = torch.load(os.path.join(input_path_1, "adapter_model.bin"))
        adapter_weights_2 = torch.load(os.path.join(input_path_2, "adapter_model.bin"))

        for param_key in adapter_weights_1.keys():
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type].from_pretrained(input_path_1)

            adapter_weights_1[param_key] = adapter_weights_1[param_key].view(
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            adapter_weights_2[param_key] = adapter_weights_2[param_key].view(
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )

            for a in tqdm(range(adapter_weights_1[param_key].size(0)), desc="Projection"):
                for b in range(adapter_weights_1[param_key].size(1)):
                    weight_1 = adapter_weights_1[param_key][a, b]
                    weight_2 = adapter_weights_2[param_key][a, b]
                    norm_weight_1 = weight_1 / torch.norm(weight_1, dim=-1, keepdim=True)
                    norm_weight_2 = weight_2 / torch.norm(weight_2, dim=-1, keepdim=True)

                    common_feature = norm_weight_1 + norm_weight_2
                    norm_common_feature = common_feature / torch.norm(common_feature, dim=-1, keepdim=True)

                    project_weight = []
                    for i in range(weight_2.size(0)):
                        project_weight.append(torch.dot(weight_2[i], norm_common_feature[i]) * norm_common_feature[i])
                    project_weight = torch.stack(project_weight)
                    
                    adapter_weights_1[param_key][a, b] = adapter_weights_1[param_key][a, b] - alpha * adapter_weights_2[param_key][a, b] + alpha * project_weight

            adapter_weights_1[param_key] = adapter_weights_1[param_key].view(peft_config.num_virtual_tokens, -1)
        
        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))
    else:
        raise NotImplementedError(peft_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_1", type=str, default="")
    parser.add_argument("--input_path_2", type=str, default="")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--method", type=str, default="ext-sub", choices=["subtraction", "addition", "ext-sub"])
    parser.add_argument("--output_path", type=str, default="")

    args = parser.parse_args()
    print(args)

    copy_folder(args.input_path_1, args.output_path, except_names=["adapter_model.bin"])

    if args.method == "subtraction":
        weight_subtraction(args.input_path_1, args.input_path_2, args.alpha, args.output_path)
    elif args.method == "addition":
        weigth_addition(args.input_path_1, args.input_path_2, args.alpha, args.output_path)
    elif args.method == "ext-sub":
        weigth_extraction_before_subtraction(args.input_path_1, args.input_path_2, args.alpha, args.output_path)
    else:
        raise NotImplementedError
    print("Processing Done.")