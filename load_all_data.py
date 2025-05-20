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

from loading_nectar import *
from loading_wildfeedback import *

def get_data(dataset_to_load, split="train", op_name="", prompt=""):
    print("Loading data..")
    if dataset_to_load == "anthropic":
        # load dataset
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
        # to pandas
        if split == "train":
            d = dataset["train"].to_pandas()
        elif split == "test":
            d = dataset["test"].to_pandas()

        data_dict = {"Number": [], "Chosen": [], "Rejected": [], "Input1": [], "Input2": [], "Dialog": []}
        ii = 0
        for i in range(len(d)):
            dialog_ = []
            if d["chosen"][i].count("Human:") <= 3:
                continue # I want only >= 4 human-assistant turn pairs 
            if d["chosen"][i].count("Human:") != d["chosen"][i].count("Assistant:"):
                continue
            if d["rejected"][i].count("Human:") != d["rejected"][i].count("Assistant:"):
                continue
            if ("\n\nHuman\n\n" in d["chosen"][i]) or ("\n\nHuman\n\n" in d["rejected"][i]):
                # very specific anomaly
                continue

            chosen = d["chosen"][i].replace("Human:", "<break> Human:").replace("Assistant:", "<break> Assistant:")
            rejected = d["rejected"][i].replace("Human:", "<break> Human:").replace("Assistant:", "<break> Assistant:")
            chosen = chosen.split("<break>")
            rejected = rejected.split("<break>")
            chosen = [ch.lstrip().rstrip() for ch in chosen]
            chosen = [ch for ch in chosen if len(ch)>0]
            rejected = [ch.lstrip().rstrip() for ch in rejected]
            rejected = [ch for ch in rejected if len(ch)>0]

            if len(chosen) != len(rejected):
                print(i, "length didn't match")
                continue
            if chosen[-1] == rejected[-1]:
                print(i)
                print("Chosen:", chosen[-1])
                print("Rejected:", rejected[-1])
                print("chosen and rejected are the same!")
                continue

            input_str1, input_str2 = deepcopy(prompt), deepcopy(prompt)
            for j in range(len(chosen)-1):
                input_str1 += "\n" + chosen[j]
                input_str2 += "\n" + chosen[j]
                dialog_.append(chosen[j])

            input_str1 += "\n" + chosen[-1].replace("Assistant:", "Assistant-1:")
            input_str1 += "\n" + rejected[-1].replace("Assistant:", "Assistant-2:")
            input_str1 += "\n\nOutput -"

            input_str2 += "\n" + rejected[-1].replace("Assistant:", "Assistant-1:")
            input_str2 += "\n" + chosen[-1].replace("Assistant:", "Assistant-2:")
            input_str2 += "\n\nOutput -"


            dialog_.append(chosen[-1].replace("Assistant:", "Assistant-1:"))
            dialog_.append(rejected[-1].replace("Assistant:", "Assistant-2:"))

            # message format
            model_input1 = [
            {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. You are a helpful assistant."},
            {"role": "user", "content": input_str1}
            ]

            model_input2 = [
            {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. You are a helpful assistant."},
            {"role": "user", "content": input_str2}
            ]

            data_dict["Number"].append(ii)
            data_dict["Input1"].append(model_input1)
            data_dict["Input2"].append(model_input2)
            data_dict["Dialog"].append("\n".join(dialog_))
            ii += 1

    elif dataset_to_load == "wildfeedback":
        data_dict = load_wildfeedback(given_prompt=prompt, condition=4, condition_type="ge")
    elif dataset_to_load == "nectar":
        data_dict_orig = load_nectar(given_prompt=prompt, condition=4, condition_type="ge",
                                    end_ind=100000)

        data_dict = {"Number": [], "Input1": [], "Input2": [], "Dialog": []}
        anthropic = get_data("anthropic", "train")

        k1 = data_dict_orig["Dialog"]
        # import pdb
        # pdb.set_trace()
        k2 = anthropic["Dialog"] #["\n".join(dd) for dd in anthropic["Dialog"]]
        l1=[]
        l2=[]
        l3=[]
        l4=[]
        l = []
        remove = []
        for i in range(len(k1)):
            if i%100==0: print("hello im at:", i)
            for j in range(len(k2)):
                if ((i,j) in l) or ((j,i) in l): continue
                s1=k1[i].split("Assistant-1:")[0].lstrip().rstrip()
                s2=k2[j].split("Assistant-1:")[0].lstrip().rstrip()
                s1_1=k1[i].split("Assistant-1:")[1].split("Assistant-2:")[0].lstrip().rstrip()
                s2_1=k2[j].split("Assistant-1:")[1].split("Assistant-2:")[0].lstrip().rstrip()
                s1_2=k1[i].split("Assistant-2:")[1].lstrip().rstrip()
                s2_2=k2[j].split("Assistant-2:")[1].lstrip().rstrip()
                if len(s1)==1 or len(s2)==1: continue
                if s1==s2:
                    l.append((i,j))
                    if s1_1 in [s2_1, s2_2] or s1_2 in [s2_1, s2_2]:
                        #print("warning!", i,j)
                        l3.append((i,j))
                    if s1_1==s2_2 and s1_2==s2_1:
                        remove.append(i)
                        print("warning!", i,j)
                    if s1_1==s2_1 and s1_2==s2_2:
                        remove.append(i)
                        print("same conv as in anthropic!", i, j)
                    if s1_1==s2_1 and s1_2==s2_2:
                        l1.append((i,j))
                        l2.extend([i,j])
                        print(i,j)

        new_id = 0
        for i in range(len(k1)):
            if i in remove: continue
            data_dict["Number"].append(new_id)
            data_dict["Input1"].append(data_dict_orig["Input1"][i])
            data_dict["Input2"].append(data_dict_orig["Input2"][i])
            data_dict["Dialog"].append(data_dict_orig["Dialog"][i])

            new_id += 1

    print(dataset_to_load, split, len(data_dict["Dialog"]))
    if len(op_name) > 0:
        pickle.dump(data_dict, open(op_name, "wb"))
    return data_dict
get_data("anthropic", "test", "all_valid_data/anthropic-test-ge4.pkl")
# get_data("wildfeedback", "train", "all_valid_data/wildfeedback-train-ge4.pkl")
# get_data("nectar", "train", "all_valid_data/nectar-train-ge4.pkl")