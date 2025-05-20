from typing import Dict, List
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
from datasets import load_dataset
from tqdm import tqdm
import pickle 

import sys

from loading_prism import *
from loading_wildfeedback import *
from loading_nectar import *

def get_args():
    parser = argparse.ArgumentParser(description='Skywork v0.2')
    parser.add_argument(
        '--dataset-test', type=str, default='anthropic-test',
        help='probably dataset names, we can get from hugging-face')
    parser.add_argument(
        '--save-path-test', type=str, default="",
        help="where to save test pickle output")
    parser.add_argument(
        '--hf_auth_token', type=str, default='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',
        help='auth token for hugging-face')
    args = parser.parse_args()
    return args

model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

correct = []
wrong = []

def get_examples(dataset_test):
    # with open("conversation.json", "r", encoding="utf-8") as file:
    #     examples = json.load(file)
    print("Loading data..")
    if dataset_test == "anthropic-test":
        # load dataset
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
        # to pandas
        d = dataset["train"].to_pandas()
        #d = dataset["test"].to_pandas()

        examples = []
        ii = 0
        for i in range(len(d)): #1000): # doing the first 1000 for now
            #if len(examples)==7000: break
            tmp = {"conversation": [], "chosen": {}, "rejected": {}}

            orig_dialog = []
            if d["chosen"][i].count("Human:") <= 3: #2:
                continue # I want only >= 3 human-assistant turn pairs 
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

            for j in range(len(chosen)-1):
                orig_dialog.append(chosen[j])
                ct_turn = chosen[j].lstrip().rstrip()
                if "Human:" in ct_turn:
                    ct = {"role": "user", "content": ct_turn.split("Human:")[1].lstrip().rstrip()}
                    tmp["conversation"].append(ct)
                elif "Assistant:" in ct_turn:
                    ct = {"role": "assistant", "content": ct_turn.split("Assistant:")[1].lstrip().rstrip()}
                    tmp["conversation"].append(ct)            

            #print(chosen[-1])
            try:
                orig_dialog.append(chosen[-1].replace("Assistant:", "Assistant-1:"))
                orig_dialog.append(rejected[-1].replace("Assistant:", "Assistant-2:"))
                tmp["chosen"] = {"role": "assistant", "content": chosen[-1].split("Assistant:")[1].lstrip().rstrip()}
                tmp["rejected"] = {"role": "assistant", "content": rejected[-1].split("Assistant:")[1].lstrip().rstrip()}
            except:
                import pdb
                pdb.set_trace()
            tmp["orig-dialog"] = "\n".join(orig_dialog).lstrip().rstrip()
            examples.append(tmp)
            ii += 1
    elif dataset_test == "prism":
        examples = load_prism(given_prompt="", condition=4, what_type="sota-rm", any_turn=0)
    elif dataset_test == "wildfeedback":
        examples = load_wildfeedback(given_prompt="", condition=4, what_type="sota-rm")
    elif dataset_test == "nectar":
        examples = load_nectar(given_prompt="", condition=4, what_type="sota-rm",
                                    end_ind=100000)


    return examples

def get_score():
    save_pickle = {}
    for example in tqdm(examples):
        conversation = example["conversation"]
        chosen_response = example["chosen"]
        rejected_response = example["rejected"]
        
        if 1: #try:
            chosen_response_tokenized = rm_tokenizer.apply_chat_template(conversation + [chosen_response], tokenize=True, return_tensors="pt").to("cuda")
            rejected_response_tokenized = rm_tokenizer.apply_chat_template(conversation + [rejected_response], tokenize=True, return_tensors="pt").to("cuda")

            score_chosen = rm(chosen_response_tokenized).logits[0][0].item()
            score_rejected = rm(rejected_response_tokenized).logits[0][0].item()
            # score_chosen = rm(conversation + [chosen_response])["score"]
            # score_rejected = rm(conversation + [rejected_response])["score"]
            
            example_with_scores = {
                "conversation": conversation,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "score_chosen": score_chosen,
                "score_rejected": score_rejected
            }

            if score_chosen > score_rejected:
                correct.append(example_with_scores)
            else:
                wrong.append(example_with_scores)

            # save as dict for pickle 
            dialog_ = example["orig-dialog"] # ""
            '''
            for cc in conversation:
                role = ""
                if cc["role"] == "user":
                    role = "Human:"
                elif cc["role"] == "assistant":
                    role = "Assistant:"
                dialog_ += role + " " + cc["content"] + "\n"
            dialog_ += "Assistant-1: " + chosen_response["content"] + "\n"
            dialog_ += "Assistant-2: " + rejected_response["content"]
            '''
            save_pickle[dialog_] = {
                "score_chosen": score_chosen,
                "score_rejected": score_rejected            
            }

        if 0: #except Exception as e:
            print(f"Error processing example: {e}")
    pickle.dump(save_pickle, open(args.save_path_test, "wb"))

def save_to_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

args = get_args()
examples = get_examples(args.dataset_test)
print("Size of dataset:", len(examples))

get_score()
save_to_jsonl("correct.json", correct)
save_to_jsonl("wrong.json", wrong)

print(f"Saved {len(correct)} correct examples to correct.json")
print(f"Saved {len(wrong)} wrong examples to wrong.json")


print(f'Final Score: {len(correct) / (len(wrong) +len(correct))}')
