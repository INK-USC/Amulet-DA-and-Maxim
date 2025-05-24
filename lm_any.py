# maxim classification
import os
import argparse
from tqdm import tqdm
from copy import deepcopy
import numpy as np

import openai
from datasets import load_dataset
import pandas as pd
import pickle
import pdb
import argparse
import torch 

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    #is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from all_prompts import *
from load_all_data import *

openai.api_key='fill-here'

def get_args():
    parser = argparse.ArgumentParser(description='Other LM')
    parser.add_argument(
        '--which-dataset', type=str, default='anthropic-train',
        help='dataset names')
    parser.add_argument(
        '--save-path', type=str, default="",
        help="where to save pickle output")
    parser.add_argument(
        '--hf_auth_token', type=str, default='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',
        help='auth token for hugging-face')
    parser.add_argument(
        '--which-model', type=str, default="gpt-4o-mini-2024-07-18",
        help="gpt-x version or HF model")
    parser.add_argument(
        '--max_new_tokens', type=int, default=256, help='')
    parser.add_argument(
        '--max_words_in_turn', type=int, default=-1, help='maximum number of words in turn')
    parser.add_argument(
        '--start_ind', type=int, default=-1, help='')
    parser.add_argument(
        '--end_ind', type=int, default=-1, help='')
    parser.add_argument(
        '--which-mode', type=str, default="da", help="da or io or w-expl")
    args = parser.parse_args()
    return args

def get_general_prompt(mode):
    if mode == "da-claude":
        prompt = dialogact_prompt_final_claude
    if mode == "da":
        prompt = dialogact_prompt_final
    if mode == "io":
        prompt = io_prompt_final
    if mode == "w-expl":
        prompt = wexpl_prompt_final
    if mode == "maxim":
        prompt = maxim_prompt_final.replace('"Final Answer"', '"Answer"')

    return prompt


def get_data(which_dataset, which_mode="da"):
    prompt = get_general_prompt(which_mode)

    data_dict = get_data(which_dataset.split("-")[0], which_dataset.split("-")[1],
                            op_name="", prompt=prompt)

    return data_dict

def run_local_model(model_input, tokenizer, model):
    inputs1 = tokenizer.apply_chat_template(
        model_input,
        add_generation_prompt=True,
        tokenize=False)
    inputs = tokenizer(inputs1, return_tensors="pt", padding=True)
    attention_mask = inputs["attention_mask"].to("cuda")

    if tokenizer.convert_tokens_to_ids("<|eot_id|>"):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = [tokenizer.eos_token_id]
    outputs = model.generate(
        #input_ids,
        inputs['input_ids'].to("cuda"),
        attention_mask=attention_mask,
        max_new_tokens=inputs["input_ids"].shape[-1]+300,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    response = outputs[0][inputs["input_ids"].shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True)    
    return output

def main():
    args = get_args()
    data_dict = get_data(args.which_dataset, args.which_mode, args.max_words_in_turn)


    config_4bit = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False)
    tokenizer = AutoTokenizer.from_pretrained(args.which_model, use_auth_token=args.hf_auth_token)
    model = AutoModelForCausalLM.from_pretrained(args.which_model, use_auth_token=args.hf_auth_token, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=config_4bit) #, attn_implementation="flash_attention_2")

    num_gpus = torch.cuda.device_count()
    print("num_gpus detected:", num_gpus)
    model.eval()

    if "Llama" in args.which_model or "gemma" in args.which_model or "Qwen" in args.which_model:
        if not tokenizer.pad_token: # Set pad token if it doesn't exist
            print("pad ttoken doesn't exist, setting to <|eot_id|>")
            tokenizer.pad_token = "<|eot_id|>" #tokenizer.eos_token 
        save_pickle = {}
        conv_ind = 0
        acc = []
        if args.start_ind == -1:
            args.start_ind = 0
        if args.end_ind == -1:
            args.end_ind = len(data_dict["Number"])
        for i in tqdm(range(args.start_ind, args.end_ind)):
            model_input1 = data_dict["Input1"][i]
            model_input2 = data_dict["Input2"][i]
            dialog_ = data_dict["Dialog"][i]
            try:
                output_dict = {}
                explanations = []
                # for _ in range(2): # just to generate twice
                model_inputs = [model_input1, model_input2]
                for ii, model_input in enumerate(model_inputs):
                    if "gemma" in args.which_model:
                        model_input = model_input[1:] # gemma cannot handle a system role
                    output = run_local_model(model_input, tokenizer, model, args.max_new_tokens)
                    #print(output)

                    if args.which_mode == "da-claude":
                        ind1 = output.find("{")
                    else:
                        ind1 = output.rfind("{")
                    ind2 = output.rfind("}")
                    output_dict_temp = eval(output[ind1:ind2+1])
                    #print(output_dict_temp)

                    assert output_dict_temp["Answer"] in ["1", "2", 1, 2]
                    if args.which_mode in ["da", "w-expl", "maxim", "da", "da-claude"]:
                        assert "Explanation" in output_dict_temp
                    if args.which_mode == "maxim":
                        for maxim in ["Quantity-1", "Quantity-2", "Quality", "Relevance-1", "Relevance-2", "Manner-1", "Manner-2", "Benevolence-1", "Benevolence-2", "Transparency-1", "Transparency-2", "Transparency-3"]:
                            assert maxim in output_dict_temp
                            assert output_dict_temp[maxim].lower() in ["1", "2", "both", "neither", 1, 2]
                    if args.which_mode == "da":
                        assert "<SEP>" in output
                        output_dict_temp["dialog-act"] = output[:ind1].lstrip().rstrip()
                    if args.which_mode == "da-claude":
                        num_keys = 0
                        for keykey in output_dict_temp:
                            if keykey in ["Answer", "Explanation"]: continue
                            assert '"Dim"' in output_dict_temp[keykey]
                            assert '"Func"' in output_dict_temp[keykey]
                            num_keys += 1
                        assert num_keys == dialog_.count("Human") + dialog_.count("Assistant")           

                    output_dict[ii] = output_dict_temp

                output_pred_1 = output_dict[0]
                output_pred_2_swapped = {"Answer": -1}
                if args.which_mode in ["w-expl", "maxim", "da", "da-claude"]:
                    output_pred_2_swapped["Explanation"] = ""
                if output_dict[1]["Answer"] == "1":
                    output_pred_2_swapped["Answer"] = "2"
                elif output_dict[1]["Answer"] == "2":
                    output_pred_2_swapped["Answer"] = "1"
                if args.which_mode in ["w-expl", "maxim", "da", "da-claude"]:
                    output_pred_2_swapped["Explanation"] = "SWAPPED!!! " + output_dict[1]["Explanation"]
                if args.which_mode == "da":
                    output_pred_2_swapped["dialog-act"] = "SWAPPED!!! " + output_dict[1]["dialog-act"]
                if args.which_mode == "da-claude":
                    for keykey in output_dict[1]:
                        if keykey in ["Answer", "Explanation"]: continue
                        output_pred_2_swapped[keykey] = output_dict[1][keykey]                    
                if args.which_mode == "maxim":
                    for maxim in ["Quantity-1", "Quantity-2", "Quality", "Relevance-1", "Relevance-2", "Manner-1", "Manner-2", "Benevolence-1", "Benevolence-2", "Transparency-1", "Transparency-2", "Transparency-3"]: 
                        if output_dict[1][maxim] in ["1", 1]:
                            output_pred_2_swapped[maxim] = "2"
                        elif output_dict[1][maxim] in ["2", 2]:
                            output_pred_2_swapped[maxim] = "1"
                        else:
                            output_pred_2_swapped[maxim] = output_dict[1][maxim]

                dialog_key = dialog_
                save_pickle[dialog_key] = {}
                save_pickle[dialog_key]["output_pred_1"] = output_pred_1
                save_pickle[dialog_key]["output_pred_2"] = output_pred_2_swapped

                if output_pred_1["Answer"]=="1" and output_pred_2_swapped["Answer"]=="1":
                    acc.append(1)
                else:
                    acc.append(0)

                if conv_ind%50==0:
                    print(len(acc), "--", np.mean(acc)*100, "%")
                    pickle.dump(save_pickle, open(args.save_path, "wb"))
                if conv_ind%1000==0:
                    print(output_pred_1)
                conv_ind += 1
                # pdb.set_trace()

            except Exception as e:
                print(e)
                print("model call failed")
        pickle.dump(save_pickle, open(args.save_path, "wb"))

if __name__=="__main__":
    main()


'''
python lm_any.py --which-dataset anthropic-test --save-path fill-here --which-model Qwen/Qwen2.5-32B-Instruct --which-mode maxim
'''



