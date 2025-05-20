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

from all_prompts import *
from load_all_data import *

# GPT-4o
openai.api_key='fill-here'
# Claude-3-5-Sonnet
import anthropic
client = anthropic.Anthropic(
    api_key="fill-here",
)


def main():
    data_dict = get_data("nectar", "train", op_name="", prompt=wexpl_prompt_final)

    print("Now running gpt-4o..")
    gpt4_model_name = "gpt-4o-2024-08-06"
    # iterating through gpt-4o
    save_dict = {"No.": [], "Dialog": []}
    save_pickle = {}
    conv_ind = 0
    acc = []
    for i in tqdm(range(len(data_dict["Number"]))):
        model_input1 = data_dict["Input1"][i]
        model_input2 = data_dict["Input2"][i]
        dialog_ = data_dict["Dialog"][i]

        try:
            output_dict = {}
            explanations = []

            model_inputs = [model_input1, model_input2]
            for ii, model_input in enumerate(model_inputs):
                output = openai.chat.completions.create(
                    model=gpt4_model_name,
                    messages=model_input,
                    temperature=0.0,
                    n=1)

                output = output.choices[0].message.content

                ind1 = output.rfind("{")
                ind2 = output.rfind("}")
                output_dict_temp = eval(output[ind1:ind2+1])

                assert output_dict_temp["Answer"] in ["1", "2"]#, "both"]
                assert "Explanation" in output_dict_temp
                output_dict[ii] = output_dict_temp

            output_pred_1 = output_dict[0]
            output_pred_2_swapped = {"Answer": -1, "Explanation": ""}
            if output_dict[1]["Answer"] == "1":
                output_pred_2_swapped["Answer"] = "2"
            elif output_dict[1]["Answer"] == "2":
                output_pred_2_swapped["Answer"] = "1"

            output_pred_2_swapped["Explanation"] = "SWAPPED!!! " + output_dict[1]["Explanation"]
            dialog_key = dialog_
            save_pickle[dialog_key] = {}
            save_pickle[dialog_key]["output_pred_1"] = output_pred_1
            save_pickle[dialog_key]["output_pred_2"] = output_pred_2_swapped

            if output_pred_1["Answer"]=="1" and output_pred_2_swapped["Answer"]=="1":
                acc.append(1)
            else:
                acc.append(0)
            if conv_ind%100==0:
                print(len(acc), "--", np.mean(acc)*100, "%")

            conv_ind += 1

        except Exception as e:
            print(e)
            chosen_1 = "api call failed"
            rejected_1 = "api call failed"
            print("api call failed")

    pickle.dump(save_pickle, open("outputs_final/nectar-train-ge4_" + gpt4_model_name + "_outputs_100k.pkl", "wb"))

    print("Now running claude..")
    claude_model_name = "claude-3-5-sonnet-20241022"
    save_dict = {"No.": [], "Dialog": []}
    save_pickle = {}
    for maxim in ["Quantity-1", "Quantity-2", "Quality", "Relevance-1", "Relevance-2", "Manner-1", "Manner-2", "Benevolence-1", "Benevolence-2", "Transparency-1", "Transparency-2", "Transparency-3"]:
        save_dict[maxim] = []
    conv_ind = 0
    acc = []
    for i in tqdm(range(len(data_dict["Number"]))):
        model_input1 = data_dict["Input1"][i]
        model_input2 = data_dict["Input2"][i]
        dialog_ = data_dict["Dialog"][i]

        try:
            maxims_dict = {}
            explanations = []
            model_inputs = [model_input1, model_input2]
            for ii, model_input in enumerate(model_inputs):
                system_prompt = "You are a helpful assistant."
                user_content = model_input[1]["content"]
                input_message = {"role": "user", "content": [{"type": "text", "text": user_content}]}

                message = client.messages.create(
                    model=claude_model_name,
                    max_tokens=4096,
                    temperature=0,
                    system=system_prompt,
                    messages=[input_message]
                )
                output_maxims = message.content[0].text
                ind1 = output_maxims.index("{")
                ind2 = output_maxims.index("}")

                maxims_dict_temp = eval(output_maxims[ind1:ind2+1])

                assert maxims_dict_temp["Final Answer"] in ["1", "2", 1, 2]#, "both"]
                assert "Explanation" in maxims_dict_temp
                for maxim in ["Quantity-1", "Quantity-2", "Quality", "Relevance-1", "Relevance-2", "Manner-1", "Manner-2", "Benevolence-1", "Benevolence-2", "Transparency-1", "Transparency-2", "Transparency-3"]:
                    assert maxim in maxims_dict_temp
                    assert maxims_dict_temp[maxim].lower() in ["1", "2", "both", "neither", 1, 2]
                maxims_dict[ii] = maxims_dict_temp

            maxim_pred_1 = maxims_dict[0]
            maxim_pred_2_swapped = maxims_dict[1]
            maxim_pred_2 = {}
            for maxim in maxim_pred_2_swapped:
                if maxim=="Explanation": 
                    maxim_pred_2["Explanation"] = "SWAPPED!!!! " + maxim_pred_2_swapped["Explanation"]
                    continue
                tmp =maxim_pred_2_swapped[maxim]
                if tmp in ["both", "neither"]:
                    tmp1 = tmp
                elif tmp == "1":
                    tmp1 = "2"
                elif tmp == "2":
                    tmp1 = "1"
                maxim_pred_2[maxim] = tmp1
            
            dialog_key = "\n".join(dialog_).lstrip().rstrip()
            save_pickle[dialog_key] = {}
            save_pickle[dialog_key]["maxim_pred_1"] = maxim_pred_1
            save_pickle[dialog_key]["maxim_pred_2"] = maxim_pred_2
            # pdb.set_trace()
            if maxim_pred_1["Final Answer"]=="1" and maxim_pred_2["Final Answer"]=="1":
                acc.append(1)
            else:
                acc.append(0)
            if conv_ind%100==0:
                print(len(acc), "--", np.mean(acc)*100, "%")

            conv_ind += 1

        except Exception as e:
            print(e)
            chosen_1 = "api call failed"
            rejected_1 = "api call failed"
            print("api call failed")

    print(len(acc), "--", np.mean(acc)*100, "%")
    pickle.dump(save_pickle, open("outputs_final/nectar-train-ge4_" + claude_model_name + "_maxims_with_result_full.pkl", "wb"))


if __name__=="__main__":
    main()
