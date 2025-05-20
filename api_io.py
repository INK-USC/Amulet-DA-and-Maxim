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
    data_dict = get_data("nectar", "train", op_name="", prompt=io_prompt_final)

    print("Now running gpt-4o..")
    gpt4_model_name = "gpt-4o-2024-08-06"
    # iterating through gpt-4o
    acc = []
    save_dict = {"No.": [], "Dialog": []}
    save_pickle = {}
    conv_ind = 0
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
                output_dict[ii] = output_dict_temp

            output_pred_1 = output_dict[0]
            output_pred_2_swapped = {"Answer": -1}
            if output_dict[1]["Answer"] == "1":
                output_pred_2_swapped["Answer"] = "2"
            elif output_dict[1]["Answer"] == "2":
                output_pred_2_swapped["Answer"] = "1"
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

    pickle.dump(save_pickle, open("outputs_final/nectar-train-ge4_" + gpt4_model_name + "_da_100k.pkl", "wb"))

    print("Now running claude..")
    claude_model_name = "claude-3-5-sonnet-20241022"

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

                output = message.content[0].text

                ind1 = output.index("{")
                ind2 = output.index("}")
                output_dict_temp = eval(output[ind1:ind2+1])
                assert output_dict_temp["Answer"] in ["1", "2"]#, "both"]
                output_dict[ii] = output_dict_temp

            output_pred_1 = output_dict[0]
            output_pred_2_swapped = {"Answer": -1}
            if output_dict[1]["Answer"] == "1":
                output_pred_2_swapped["Answer"] = "2"
            elif output_dict[1]["Answer"] == "2":
                output_pred_2_swapped["Answer"] = "1"

            
            dialog_key = "\n".join(dialog_).lstrip().rstrip()
            save_pickle[dialog_key] = {}
            save_pickle[dialog_key]["output_pred_1"] = output_pred_1
            save_pickle[dialog_key]["output_pred_2"] = output_pred_2_swapped
            # pdb.set_trace()
            if output_pred_1["Answer"]=="1" and output_pred_2_swapped["Answer"]=="1":
                acc.append(1)
            else:
                acc.append(0)
            if conv_ind%50==0:
                print(len(acc), "--", np.mean(acc)*100, "%")

            conv_ind += 1

        except:
            chosen_1 = "api call failed"
            rejected_1 = "api call failed"
            print("api call failed")

    print(len(acc), "--", np.mean(acc)*100, "%")
    pickle.dump(save_pickle, open("outputs_final/nectar-train-ge4_" + mode + "_" + claude_model_name + "_io_full.pkl", "wb"))



if __name__=="__main__":
    main()
