# maxim classification
from copy import deepcopy
from datasets import load_dataset
import pandas as pd
import pickle
from collections import defaultdict
import pdb

ultrarm_template = """{conversation}

{completion}"""

def get_valid_convs(d, condition=4, condition_type="ge"):
    l1 = []
    for i in range(len(d)):
        conversations = d['conversations'][i]
        chosen = d['chosen'][i]
        rejected = d['rejected'][i]

        # getting the conv
        dialog = []
        break_flag = 0
        for conv in conversations:
            text = ''
            tmpp = {}
            if conv['from'] == 'gpt':
                text += "Assistant: " 
                tmpp["role"] = "assistant"
            else:
                text += "Human: " 
                tmpp["role"] = "user"
            if len(conv["value"].split()) > 300:
                break_flag = 1 
            text += conv['value']
            tmpp["content"] = conv["value"]
            dialog.append(text)
        if break_flag:
            continue

        dialog_running = "\n".join(dialog)

        if dialog_running.count("Human:") != (dialog_running.count("Assistant:")+1):
            continue
        if condition_type == "ge":
            condition_flag = dialog_running.count("Human:") >= condition
        elif condition_type == "le":
            condition_flag = dialog_running.count("Human:") <= condition
        if not condition_flag:
            continue
        l1.append(dialog_running)

    l2 = []
    tree = defaultdict(list) # can't believe I've resorted to trees
    reverse_tree = {}
    for i in range(len(l1)):
        for j in range(len(l1)):
            if i == j: continue
            if (l1[i] in l1[j]) or (l1[j] in l1[i]): 
                if i > j:
                    w1 = j
                    w2 = i
                else:
                    w1 = i
                    w2 = j

                if w2 in reverse_tree:
                    w1 = reverse_tree[w2]
                if w1 in tree:
                    if w2 in tree[w1]:
                        continue
                    else:
                        tree[w1].append(w2)
                        reverse_tree[w2] = w1
                else:
                    tree[w1].append(w2)
                    reverse_tree[w2] = w1

    final_convs = []
    f=[]
    for kk in tree:
        values = tree[kk]
        ind, length = kk, len(l1[kk])
        for vv in range(len(values)):
            if len(l1[values[vv]]) > length:
                ind, length = values[vv], len(l1[values[vv]])
        final_convs.append(l1[ind])
        f.append(ind)
    assert len(final_convs) == len(set(final_convs))
    return l1, tree, final_convs

def load_wildfeedback(given_prompt, condition, what_type="usual", condition_type="ge", max_num_words=300):
    # load dataset
    print("hi")
    dataset = load_dataset("microsoft/WildFeedback", "wildfeedback")
    # to pandas
    d = dataset["train"].to_pandas()

    #l1, tree, valid_convs = get_valid_convs(d, condition, condition_type)

    data_dict = {"Number": [], "Input1": [], "Input2": [], "Dialog": []}
    data_dict_ultrarm = {"Number": [], "input_chosen": [], "input_rejected": [], "Dialog": []}
    examples = []


    ii = 0
    bad_count = 0
    for i in range(len(d)):
        conversations = d['conversations'][i]
        chosen = d['chosen'][i]
        rejected = d['rejected'][i]

        # getting the conv
        dialog_cr1, dialog_cr2 = [], []
        dialog_running_list = []
        break_flag = 0
        for conv in conversations:
            text = ''
            tmpp = {}
            if conv['from'] == 'gpt':
                text += "Assistant: " 
                tmpp["role"] = "assistant"
            else:
                text += "Human: " 
                tmpp["role"] = "user"
            if len(conv["value"].split()) > max_num_words:
                break_flag = 1 
            text += conv['value']
            tmpp["content"] = conv["value"]
            dialog_cr1.append(text)
            dialog_cr2.append(text)
            dialog_running_list.append(tmpp)
        if break_flag:
            continue
        #if "\n".join(dialog_cr1) not in valid_convs:
        #    continue
        if d['chosen'][i]['value'] == d['rejected'][i]['value']:
            print(i)
            print("Chosen:", d['chosen'][i]['value'])
            print("Rejected:", d['rejected'][i]['value'])
            print("chosen and rejected are the same!")
            continue            
        dialog_cr1.append('Assistant-1: ' + d['chosen'][i]['value'])
        dialog_cr1.append('Assistant-2: ' + d['rejected'][i]['value'])
        dialog_cr2.append('Assistant-1: ' + d['rejected'][i]['value'])
        dialog_cr2.append('Assistant-2: ' + d['chosen'][i]['value'])        
        dialog_running = "\n".join(dialog_cr1)

        if dialog_running.count("Human:") != (dialog_running.count("Assistant:")+1):
            continue

        if condition_type == "ge":
            condition_flag = dialog_running.count("Human:") >= condition
        elif condition_type == "le":
            condition_flag = dialog_running.count("Human:") <= condition

        if not condition_flag:
            continue


        # for GPT-4-variant prompting
        input_str1, input_str2 = deepcopy(given_prompt), deepcopy(given_prompt)

        input_str1 += "\n".join(dialog_cr1) + "\n\nOutput -"
        input_str2 += "\n".join(dialog_cr2) + "\n\nOutput -"

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
        data_dict["Dialog"].append(dialog_running)

        # SOTA RM
        tmp_chosen = {"role": "assistant", "content": d['chosen'][i]['value']}
        tmp_rejected = {"role": "assistant", "content": d['rejected'][i]['value']}
        tmp = {"conversation": deepcopy(dialog_running_list),
                    "chosen": tmp_chosen, "rejected": tmp_rejected}
        tmp["orig-dialog"] = dialog_running
        examples.append(tmp)

        # for UltraRM
        ip1, ip2 = deepcopy(ultrarm_template), deepcopy(ultrarm_template)
        model_input_chosen = ip1.format(conversation=dialog_running,
            completion=d['chosen'][i]['value'])
        model_input_rejected = ip2.format(conversation=dialog_running,
            completion=d['rejected'][i]['value'])
        data_dict_ultrarm["Number"].append(ii)
        data_dict_ultrarm["input_chosen"].append(model_input_chosen)
        data_dict_ultrarm["input_rejected"].append(model_input_rejected)
        data_dict_ultrarm["Dialog"].append(dialog_running)                

        ii += 1

    print("Total length:", len(data_dict["Number"]))

    # great now I just need to run this with GPT-4o! 
    if what_type == "usual":
        return data_dict
    elif what_type == "sota-rm":
        return examples
    elif what_type == "ultrarm":
        return data_dict_ultrarm


# a=load_wildfeedback("", condition=4, condition_type="ge")
# print(len(a["Number"]))
