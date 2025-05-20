import json
import torch
import torch
import torch.nn as nn
from transformers import LlamaPreTrainedModel, LlamaModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import List, Optional

import argparse
import pickle
from tqdm import tqdm
import sys

from loading_wildfeedback import *
from loading_nectar import *

def get_args():
    parser = argparse.ArgumentParser(description='INF-ORM')
    parser.add_argument(
        '--dataset-test', type=str, default='anthropic-test',
        help='probably dataset names, we can get from hugging-face')
    parser.add_argument(
        '--save-path-test', type=str, default="",
        help="where to save test pickle output")
    parser.add_argument(
        '--hf_auth_token', type=str, default='fill-here',
        help='auth token for hugging-face')
    args = parser.parse_args()
    return args


class INFORMForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    

correct = []
wrong = []

model_name = "infly/INF-ORM-Llama3.1-70B"
orm = INFORMForSequenceClassification.from_pretrained(
    model_name,
    #torch_dtype=torch.bfloat16,
    device_map="auto",#{"": "cuda"}, #device_map="auto",
    #attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)


def get_examples(dataset_test):
    # with open("conversation.json", "r", encoding="utf-8") as file:
    #     examples = json.load(file)  
    # return examples
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
        
        try:
            conv_chosen = conversation + [chosen_response]
            conv_rejected = conversation + [rejected_response]
            #print(conv_chosen)
            conv_chosen_tokenized = tokenizer.apply_chat_template(conv_chosen, tokenize=True, return_tensors="pt").to("cuda")
            conv_rejected_tokenized = tokenizer.apply_chat_template(conv_rejected, tokenize=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                score_chosen = orm(conv_chosen_tokenized).logits[0][0].item()
                score_rejected = orm(conv_rejected_tokenized).logits[0][0].item()
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
            dialog_ = ""
            for cc in conversation:
                role = ""
                if cc["role"] == "user":
                    role = "Human:"
                elif cc["role"] == "assistant":
                    role = "Assistant:"
                dialog_ += role + " " + cc["content"] + "\n"
            dialog_ += "Assistant-1: " + chosen_response["content"] + "\n"
            dialog_ += "Assistant-2: " + rejected_response["content"]
            save_pickle[dialog_] = {
                "score_chosen": score_chosen,
                "score_rejected": score_rejected            
            }

        except Exception as e:
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
