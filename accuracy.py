import numpy as np
import pickle
import json 

# from load_all_data import get_data

condition = 4

length_condition_flag = "no" # less, more and no
length_condition = 500
newline_condition_flag = "no" # less, more and no
newline_condition = 7
# dataset = "wildfeedback"
# dataset = "nectar"
dataset = "anthropic"

# split = "train"
split = "test"

model_name = "gpt-4o-2024-08-06"
# model_name = "qwen2.5-32B-Instruct"
# model_name = "gemma-2-27b-it"
# model_name = "claude-3-5-sonnet-20241022"

# data_dict = get_data(dataset, split)
valid_datapoints = pickle.load(open("all_valid_data/" + dataset + "-" + split + ".pkl", "rb"))
print("Number of valid_datapoints:", len(valid_datapoints["Dialog"]))
use_valid = 1

if dataset == "wildfeedback":
    model_name = "gpt-4o-2024-08-06"
    mode = "-ge4"

    io_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/wildfeedback-" + split + "-io-final_all.pkl", "rb"))
    k0=list(io_pred.keys())
    baseline_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/wildfeedback-" + split + "-w-expl-final_all.pkl", "rb"))
    k1=list(baseline_pred.keys())
    da_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/wildfeedback-" + split + "-da-final_all.pkl", "rb"))
    k2=list(da_pred.keys())

    maxim_pairwise_with_result_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/wildfeedback-" + split + "-maxim-final_all.pkl", "rb"))
    k3=list(maxim_pairwise_with_result_pred.keys())
    print(len(k0), len(k1), len(k2), len(k3))

    skywork = pickle.load(open("final_outputs/Skywork/wildfeedback-"+ split + "-ge4_Skywork-Reward-Llama-3.1-8B-v0.2_full.pkl", "rb"))
    k6 = list(skywork.keys())
    inform = pickle.load(open("final_outputs/INF-ORM/wildfeedback-"+ split + "-INFORM_full.pkl", "rb"))
    k7 = list(inform.keys())
    qrm = pickle.load(open("final_outputs/QRM/wildfeedback-"+ split + "-ge4_QRM_full.pkl", "rb"))
    k8 = list(qrm.keys())

    print(len(k4), len(k5), len(k6), len(k7), len(k8))

if dataset == "nectar":
    model_name = "gpt-4o-2024-08-06"
    mode = "-ge4"
    io_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/nectar-" + split + "-io-final_all.pkl", "rb"))
    k0=list(io_pred.keys())
    baseline_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/nectar-" + split + "-w-expl-final_all.pkl", "rb"))
    k1=list(baseline_pred.keys())
    da_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/nectar-" + split + "-da-final_all.pkl", "rb"))
    k2=list(da_pred.keys())

    maxim_pairwise_with_result_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/nectar-" + split + "-maxim-final_all.pkl", "rb"))
    k3=list(maxim_pairwise_with_result_pred.keys())
    print(len(k0), len(k1), len(k2), len(k3))

    skywork = pickle.load(open("final_outputs/Skywork/nectar-"+ split + "-ge4_Skywork-Reward-Llama-3.1-8B-v0.2_full.pkl", "rb"))
    k6 = list(skywork.keys())
    inform = pickle.load(open("final_outputs/INF-ORM/nectar-"+ split + "-INFORM_full.pkl", "rb"))
    k7 = list(inform.keys())
    qrm = pickle.load(open("final_outputs/QRM/nectar-"+ split + "-ge4_QRM_full.pkl", "rb"))
    k8 = list(qrm.keys())

    print(len(k4), len(k5), len(k6), len(k7), len(k8))


if dataset == "anthropic" and model_name == "gpt-4o-2024-08-06":
    mode = "-ge4"
    io_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/anthropic-help-" + split + "-io-final_all.pkl", "rb"))
    k0=list(io_pred.keys())
    baseline_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/anthropic-help-" + split + "-w-expl-final_all.pkl", "rb"))
    k1=list(baseline_pred.keys())
    da_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/anthropic-help-" + split + "-da-final_all.pkl", "rb"))
    k2=list(da_pred.keys())

    maxim_pairwise_with_result_pred=pickle.load(open("final_outputs/outputs_gpt4o_final/anthropic-help-" + split + "-maxim-final_all.pkl", "rb"))
    k3=list(maxim_pairwise_with_result_pred.keys())
    print(len(k0), len(k1), len(k2), len(k3))

    skywork = pickle.load(open("final_outputs/Skywork/anthropic-help-"+ split + "-ge4_Skywork-Reward-Llama-3.1-8B-v0.2_full.pkl", "rb"))
    k6 = list(skywork.keys())
    inform = pickle.load(open("final_outputs/INF-ORM/anthropic-help-"+ split + "-INFORM_full.pkl", "rb"))
    k7 = list(inform.keys())
    qrm = pickle.load(open("final_outputs/QRM/anthropic-help-"+ split + "-ge4_QRM_full.pkl", "rb"))
    k8 = list(qrm.keys())

    print(len(k4), len(k5), len(k6), len(k7), len(k8))


if dataset == "anthropic" and model_name == "qwen2.5-32B-Instruct":
    mode = "-ge4"
    
    io_pred=pickle.load(open("final_outputs/outputs_qwen_final/anthropic-help-" + split + "-io-final_all.pkl", "rb"))
    k0=list(io_pred.keys())
    baseline_pred=pickle.load(open("final_outputs/outputs_qwen_final/anthropic-help-" + split + "-w-expl-final_all.pkl", "rb"))
    k1=list(baseline_pred.keys())
    da_pred=pickle.load(open("final_outputs/outputs_qwen_final/anthropic-help-" + split + "-da-final_all.pkl", "rb"))
    k2=list(da_pred.keys())

    maxim_pairwise_with_result_pred=pickle.load(open("final_outputs/outputs_qwen_final/anthropic-help-" + split + "-maxim-final_all.pkl", "rb"))
    k3=list(maxim_pairwise_with_result_pred.keys())
    print(len(k0), len(k1), len(k2), len(k3))

    skywork = pickle.load(open("final_outputs/Skywork/anthropic-help-"+ split + "-ge4_Skywork-Reward-Llama-3.1-8B-v0.2_full.pkl", "rb"))
    k6 = list(skywork.keys())
    inform = pickle.load(open("final_outputs/INF-ORM/anthropic-help-"+ split + "-INFORM_full.pkl", "rb"))
    k7 = list(inform.keys())
    qrm = pickle.load(open("final_outputs/QRM/anthropic-help-"+ split + "-ge4_QRM_full.pkl", "rb"))
    k8 = list(qrm.keys())

    print(len(k4), len(k5), len(k6), len(k7), len(k8))

if dataset == "anthropic" and model_name == "claude-3-5-sonnet-20241022":
    mode = "-ge4"
    
    io_pred=pickle.load(open("final_outputs/outputs_claude_final/anthropic-help-" + split + "-io-final_all.pkl", "rb"))
    k0=list(io_pred.keys())
    baseline_pred=pickle.load(open("final_outputs/outputs_claude_final/anthropic-help-" + split + "-w-expl-final_all.pkl", "rb"))
    k1=list(baseline_pred.keys())
    da_pred=pickle.load(open("final_outputs/outputs_claude_final/anthropic-help-" + split + "-da-final_all.pkl", "rb"))
    k2=list(da_pred.keys())

    maxim_pairwise_with_result_pred=pickle.load(open("final_outputs/outputs_claude_final/anthropic-help-" + split + "-maxim-final_all.pkl", "rb"))
    k3=list(maxim_pairwise_with_result_pred.keys())
    print(len(k0), len(k1), len(k2), len(k3))

    skywork = pickle.load(open("final_outputs/Skywork/anthropic-help-"+ split + "-ge4_Skywork-Reward-Llama-3.1-8B-v0.2_full.pkl", "rb"))
    k6 = list(skywork.keys())
    inform = pickle.load(open("final_outputs/INF-ORM/anthropic-help-"+ split + "-INFORM_full.pkl", "rb"))
    k7 = list(inform.keys())
    qrm = pickle.load(open("final_outputs/QRM/anthropic-help-"+ split + "-ge4_QRM_full.pkl", "rb"))
    k8 = list(qrm.keys())

    print(len(k4), len(k5), len(k6), len(k7), len(k8))

if use_valid:
    k = valid_datapoints["Dialog"]

op_str = []
op_str_tie = []

# IO accuracy
# k=list(io_pred.keys())
acc_io_pred=[]
acc_position_bias_io_pred = []
acc_second_position_bias_io_pred = []
dict_io_pred = {}
win_lose_tie_io_pred = {"win": 0, "tie":0, "lose": 0}
for kk in k:
    if kk.count("Human:") < condition:
        continue
    if use_valid and kk not in io_pred:
        acc_io_pred.append(0)
        acc_position_bias_io_pred.append(0)
        acc_second_position_bias_io_pred.append(0)
        continue

    m1 = io_pred[kk]["output_pred_1"]["Answer"]
    m2 = io_pred[kk]["output_pred_2"]["Answer"]

    if (m1=="1" and m2=="1"): # or (m1=="both" and m2=="both"):
        acc_io_pred.append(1)
        dict_io_pred[kk] = 1
    else:
        acc_io_pred.append(0)
        dict_io_pred[kk] = 0


    if m1=="1" and m2=="1":
        win_lose_tie_io_pred["win"] += 1
    if m1=="2" and m2=="2":
        win_lose_tie_io_pred["lose"] += 1
    if (m1=="1" and m2=="2") or (m1=="2" and m2=="1"):
        win_lose_tie_io_pred["tie"] += 1

    # with position bias
    if m1=="1":
        acc_position_bias_io_pred.append(1)
    else:
        acc_position_bias_io_pred.append(0)

    if m2=="2":
        acc_second_position_bias_io_pred.append(1)
    else:
        acc_second_position_bias_io_pred.append(0)
print("\nI/O accuracy: ")
print("Accuracy accounting for position bias:", len(acc_io_pred), 100*np.mean(acc_io_pred))
print("Accuracy with position bias:", len(acc_position_bias_io_pred), 100*np.mean(acc_position_bias_io_pred))
# print("Accuracy with second position bias:", len(acc_second_position_bias_io_pred), 100*np.mean(acc_second_position_bias_io_pred))
op_str.append(round(100*np.mean(acc_io_pred),1))
tot = sum(win_lose_tie_io_pred.values())
win_lose_tie_io_pred["win"] = round(100*win_lose_tie_io_pred["win"]/tot,1)
win_lose_tie_io_pred["lose"] = round(100*win_lose_tie_io_pred["lose"]/tot,1)
win_lose_tie_io_pred["tie"] = round(100*win_lose_tie_io_pred["tie"]/tot,1)
op_str_tie.append("I/O: win / tie / loss - " + str(win_lose_tie_io_pred["win"]) + " / " + str(win_lose_tie_io_pred["tie"]) + " / " + str(win_lose_tie_io_pred["lose"]))
# op_str_tie.append("I/O: " + str(win_lose_tie_io_pred))

# Baseline accuracy
acc_baseline_pred=[]
acc_position_bias_baseline_pred = []
dict_baseline_pred = {}
win_lose_tie_baseline_pred = {"win": 0, "tie":0, "lose": 0}
for kk in k:
    if kk.count("Human:") < condition:
        continue
    if use_valid and kk not in baseline_pred:
        acc_baseline_pred.append(0)
        acc_position_bias_baseline_pred.append(0)
        continue
    asst2 = kk.split("Assistant-2:")[1].lstrip().rstrip()
    asst1 = kk.split("Assistant-1:")[1].split("Assistant-2:")[0].lstrip().rstrip()
    if length_condition_flag != "no":
        if length_condition_flag == "less" and len(asst1) > length_condition:
            continue
        elif length_condition_flag == "more" and len(asst1) < length_condition:
            continue
    if newline_condition_flag != "no":
        if newline_condition_flag == "less" and asst1.count("\n") > newline_condition:
            continue
        elif newline_condition_flag == "more" and asst1.count("\n") < newline_condition:
            continue
    m1 = baseline_pred[kk]["output_pred_1"]["Answer"]
    m2 = baseline_pred[kk]["output_pred_2"]["Answer"]

    if (m1=="1" and m2=="1"): # or (m1=="both" and m2=="both"):
        acc_baseline_pred.append(1)
        dict_baseline_pred[kk] = 1
    else:
        acc_baseline_pred.append(0)
        dict_baseline_pred[kk] = 0

    if m1=="1" and m2=="1":
        win_lose_tie_baseline_pred["win"] += 1
    if m1=="2" and m2=="2":
        win_lose_tie_baseline_pred["lose"] += 1
    if (m1=="1" and m2=="2") or (m1=="2" and m2=="1"):
        win_lose_tie_baseline_pred["tie"] += 1


    # with position bias
    if m1=="1":
        acc_position_bias_baseline_pred.append(1)
    else:
        acc_position_bias_baseline_pred.append(0)
print("\nW-expl accuracy: ")
print("Accuracy accounting for position bias:", len(acc_baseline_pred), 100*np.mean(acc_baseline_pred))
print("Accuracy with position bias:", len(acc_position_bias_baseline_pred), 100*np.mean(acc_position_bias_baseline_pred))
op_str.append(round(100*np.mean(acc_baseline_pred),1))
tot = sum(win_lose_tie_baseline_pred.values())
win_lose_tie_baseline_pred["win"] = round(100*win_lose_tie_baseline_pred["win"]/tot,1)
win_lose_tie_baseline_pred["lose"] = round(100*win_lose_tie_baseline_pred["lose"]/tot,1)
win_lose_tie_baseline_pred["tie"] = round(100*win_lose_tie_baseline_pred["tie"]/tot,1)
op_str_tie.append("W-Expl: win / tie / loss - " + str(win_lose_tie_baseline_pred["win"]) + " / " + str(win_lose_tie_baseline_pred["tie"]) + " / " + str(win_lose_tie_baseline_pred["lose"]))
# op_str_tie.append("W-Expl: " + str(win_lose_tie_baseline_pred))

# Dialog Act based accuracy
acc_da_pred=[]
acc_position_bias_da_pred = []
dict_da_pred = {}
win_lose_tie_da_pred = {"win": 0, "tie":0, "lose": 0}
for kk in k:
    if kk.count("Human:") < condition:
        continue
    if use_valid and kk not in da_pred:
        acc_da_pred.append(0)
        acc_position_bias_da_pred.append(0)
        continue
    asst2 = kk.split("Assistant-2:")[1].lstrip().rstrip()
    asst1 = kk.split("Assistant-1:")[1].split("Assistant-2:")[0].lstrip().rstrip()
    if length_condition_flag != "no":
        if length_condition_flag == "less" and len(asst1) > length_condition:
            continue
        elif length_condition_flag == "more" and len(asst1) < length_condition:
            continue
    if newline_condition_flag != "no":
        if newline_condition_flag == "less" and asst1.count("\n") > newline_condition:
            continue
        elif newline_condition_flag == "more" and asst1.count("\n") < newline_condition:
            continue
    m1 = da_pred[kk]["output_pred_1"]["Answer"]
    m2 = da_pred[kk]["output_pred_2"]["Answer"]

    if (m1=="1" and m2=="1"):# or (m1=="both" and m2=="both"):
        acc_da_pred.append(1)
        dict_da_pred[kk] = 1
    else:
        acc_da_pred.append(0)
        dict_da_pred[kk] = 0

    if m1=="1" and m2=="1":
        win_lose_tie_da_pred["win"] += 1
    if m1=="2" and m2=="2":
        win_lose_tie_da_pred["lose"] += 1
    if (m1=="1" and m2=="2") or (m1=="2" and m2=="1"):
        win_lose_tie_da_pred["tie"] += 1


    # with position bias
    if m1=="1":
        acc_position_bias_da_pred.append(1)
    else:
        acc_position_bias_da_pred.append(0)
print("\nDialog Act method's accuracy: ")
print("Accuracy accounting for position bias:", len(acc_da_pred), 100*np.mean(acc_da_pred))
print("Accuracy with position bias:", len(acc_position_bias_da_pred), 100*np.mean(acc_position_bias_da_pred))
op_str.append(round(100*np.mean(acc_da_pred),1))
tot = sum(win_lose_tie_da_pred.values())
win_lose_tie_da_pred["win"] = round(100*win_lose_tie_da_pred["win"]/tot,1)
win_lose_tie_da_pred["lose"] = round(100*win_lose_tie_da_pred["lose"]/tot,1)
win_lose_tie_da_pred["tie"] = round(100*win_lose_tie_da_pred["tie"]/tot,1)
op_str_tie.append("DA: win / tie / loss - " + str(win_lose_tie_da_pred["win"]) + " / " + str(win_lose_tie_da_pred["tie"]) + " / " + str(win_lose_tie_da_pred["lose"]))
# op_str_tie.append("DA: " + str(win_lose_tie_da_pred))

# maxim pairwise with result
acc_maxim_pairwise_with_result=[]
acc_position_bias_maxim_pairwise_with_result = []
dict_maxim_pairwise_with_result = {}
maxim_correct = []
win_lose_tie_maxim_pred = {"win": 0, "tie":0, "lose": 0}
for kk in k:
    if kk.count("Human:") < condition:
        continue
    if use_valid and kk not in maxim_pairwise_with_result_pred:
        acc_maxim_pairwise_with_result.append(0)
        acc_position_bias_maxim_pairwise_with_result.append(0)
        continue

    # print(maxim_pairwise_with_result_pred[kk])
    if "maxim_pred_1" in maxim_pairwise_with_result_pred[kk]:
        m1 = maxim_pairwise_with_result_pred[kk]["maxim_pred_1"]["Final Answer"]
        m2 = maxim_pairwise_with_result_pred[kk]["maxim_pred_2"]["Final Answer"]
    else:
        if "Final Answer" in maxim_pairwise_with_result_pred[kk]["output_pred_1"]:
            m1 = maxim_pairwise_with_result_pred[kk]["output_pred_1"]["Final Answer"]
            m2 = maxim_pairwise_with_result_pred[kk]["output_pred_2"]["Final Answer"]
        else:
            m1 = maxim_pairwise_with_result_pred[kk]["output_pred_1"]["Answer"]
            m2 = maxim_pairwise_with_result_pred[kk]["output_pred_2"]["Answer"]

    if (m1=="1" and m2=="1"): # or (m1=="both" and m2=="both"): # or (m1=="1" and m2=="both") or (m1=="both" and m2=="1"):
        # if (m1 in ["1", "both"]) and (m2 in ["1", "both"]):
        acc_maxim_pairwise_with_result.append(1)
        dict_maxim_pairwise_with_result[kk] = 1
        maxim_correct.append(kk)
    else:
        acc_maxim_pairwise_with_result.append(0)
        dict_maxim_pairwise_with_result[kk] = 0

    if m1=="1" and m2=="1":
        win_lose_tie_maxim_pred["win"] += 1
    if m1=="2" and m2=="2":
        win_lose_tie_maxim_pred["lose"] += 1
    if (m1=="1" and m2=="2") or (m1=="2" and m2=="1"):
        win_lose_tie_maxim_pred["tie"] += 1


    # with position bias
    if m1=="1":
        acc_position_bias_maxim_pairwise_with_result.append(1)
    else:
        acc_position_bias_maxim_pairwise_with_result.append(0)

    '''
    consistent_maxims = []
    for maxx in m1:
        if m1[maxx]==m2[maxx]:
            consistent_maxims.append(m1[maxx])
    cnt1 = consistent_maxims.count("1")
    cnt2 = consistent_maxims.count("2")
    cntboth = consistent_maxims.count("both")
    cntnei = consistent_maxims.count("neither")
    if cnt1 > cnt2:
        acc_maxim_pairwise_with_result.append(1)
        dict_maxim_pairwise_with_result[kk] = 1
    else:
        acc_maxim_pairwise_with_result.append(0)
        dict_maxim_pairwise_with_result[kk] = 0

    # with position bias
    pos_bias_maxims = list(m1.values())
    cnt1 = pos_bias_maxims.count("1")
    cnt2 = pos_bias_maxims.count("2")
    cntboth = pos_bias_maxims.count("both")
    cntnei = pos_bias_maxims.count("neither")
    if cnt1 > cnt2:
        acc_position_bias_maxim_pairwise_with_result.append(1)
    else:
        acc_position_bias_maxim_pairwise_with_result.append(0)
    '''
print("\nMaxim pairwise with result accuracy:")
print("Accuracy accounting for position bias:", len(acc_maxim_pairwise_with_result), 100*np.mean(acc_maxim_pairwise_with_result))
print("Accuracy with position bias:", len(acc_position_bias_maxim_pairwise_with_result), 100*np.mean(acc_position_bias_maxim_pairwise_with_result))
op_str.append(round(100*np.mean(acc_maxim_pairwise_with_result),1))
tot = sum(win_lose_tie_maxim_pred.values())
win_lose_tie_maxim_pred["win"] = round(100*win_lose_tie_maxim_pred["win"]/tot,1)
win_lose_tie_maxim_pred["lose"] = round(100*win_lose_tie_maxim_pred["lose"]/tot,1)
win_lose_tie_maxim_pred["tie"] = round(100*win_lose_tie_maxim_pred["tie"]/tot,1)
op_str_tie.append("Maxim: win / tie / loss - " + str(win_lose_tie_maxim_pred["win"]) + " / " + str(win_lose_tie_maxim_pred["tie"]) + " / " + str(win_lose_tie_maxim_pred["lose"]))
# op_str_tie.append("Maxim: " + str(win_lose_tie_maxim_pred))

# The Ensembles
# Logic-3 aka DA then maxim
# Logic-4 aka DA + maxim
logic3_acc = []
logic4_acc = []
logic5_acc = []
io_then_da = []
io_then_da_then_maxim = []
da_then_maxim_then_io = []
maxim_then_da_then_io = []
da_then_maxim_then_wexpl = []
maxim_then_da_then_wexpl = []
win_lose_tie_da_then_maxim = {"win": 0, "tie":0, "lose": 0}
win_lose_tie_da_then_maxim_then_wexpl = {"win": 0, "tie":0, "lose": 0}
# here my logic is that
# if a certain data point does not have a judgment by any one of the judges
# we can still use the output by another judge to judge it, ASSUMING this other judge gives a
# result that is same across the two votes
for kk in k:
    if kk.count("Human:") < condition:
        continue

    if use_valid and kk not in maxim_pairwise_with_result_pred:
        m1 = -1
        m2 = -2
    else:
        if "maxim_pred_1" in maxim_pairwise_with_result_pred[kk]:
            m1 = maxim_pairwise_with_result_pred[kk]["maxim_pred_1"]["Final Answer"]
            m2 = maxim_pairwise_with_result_pred[kk]["maxim_pred_2"]["Final Answer"]
        else:
            if "Final Answer" in maxim_pairwise_with_result_pred[kk]["output_pred_1"]:
                m1 = maxim_pairwise_with_result_pred[kk]["output_pred_1"]["Final Answer"]
                m2 = maxim_pairwise_with_result_pred[kk]["output_pred_2"]["Final Answer"]
            else:
                m1 = maxim_pairwise_with_result_pred[kk]["output_pred_1"]["Answer"]
                m2 = maxim_pairwise_with_result_pred[kk]["output_pred_2"]["Answer"]

    if use_valid and kk not in da_pred:
        da1 = -1
        da2 = -2 # basically, we have no signal from the da method
    else:
        da1 = da_pred[kk]["output_pred_1"]["Answer"]
        da2 = da_pred[kk]["output_pred_2"]["Answer"]

    if use_valid and kk not in io_pred:
        io1 = -1
        io2 = -2
    else:
        io1 = io_pred[kk]["output_pred_1"]["Answer"]
        io2 = io_pred[kk]["output_pred_2"]["Answer"]

    if use_valid and kk not in baseline_pred:
        we1 = -1
        we2 = -2
    else:
        we1 = baseline_pred[kk]["output_pred_1"]["Answer"]
        we2 = baseline_pred[kk]["output_pred_2"]["Answer"]

    if da1 == da2:
        if da1 == "1":
            logic3_acc.append(1)
            win_lose_tie_da_then_maxim["win"] += 1
        else: 
            logic3_acc.append(0)
            win_lose_tie_da_then_maxim["lose"] += 1
    else:
        if m1 == "1" and m2 == "1":
            logic3_acc.append(1)
            win_lose_tie_da_then_maxim["win"] += 1
        else:
            logic3_acc.append(0)
            if m1=="2" and m2=="2":
                win_lose_tie_da_then_maxim["lose"] += 1
            else:
                win_lose_tie_da_then_maxim["tie"] += 1

    logic4_votes = [da1, da2, m1, m2]
    if logic4_votes.count("1") >= 3:
        logic4_acc.append(1)
    else:
        logic4_acc.append(0)

    if m1 == m2:
        if m1 == "1":
            logic5_acc.append(1)
        else:
            logic5_acc.append(0)
    else:
        if da1 == "1" and da2 == "1":
            logic5_acc.append(1)
        else:
            logic5_acc.append(0)

    if io1 == io2:
        if io1 == "1":
            io_then_da.append(1)
        else:
            io_then_da.append(0)
    else:
        if da1 == "1" and da2 == "1":
            io_then_da.append(1)
        else:
            io_then_da.append(0)

    if io1 == io2:
        if io1 == "1":
            io_then_da_then_maxim.append(1)
        else:
            io_then_da_then_maxim.append(0)
    else:
        if da1 == da2:
            if da1 == "1":
                io_then_da_then_maxim.append(1)
            else: 
                io_then_da_then_maxim.append(0)
        else:
            if m1 == "1" and m2 == "1":
                io_then_da_then_maxim.append(1)
            else:
                io_then_da_then_maxim.append(0)

    if da1 == da2:
        if da1 == "1":
            da_then_maxim_then_io.append(1)
        else: 
            da_then_maxim_then_io.append(0)
    else:
        if m1 == m2:
            if m1 == "1":
                da_then_maxim_then_io.append(1)
            else:
                da_then_maxim_then_io.append(0)
        else:
            if io1 == "1" and io2 == "1":
                da_then_maxim_then_io.append(1)
            else:
                da_then_maxim_then_io.append(0)
    
    if da1 == da2:
        if da1 == "1":
            da_then_maxim_then_wexpl.append(1)
            win_lose_tie_da_then_maxim_then_wexpl["win"] += 1
        else: 
            da_then_maxim_then_wexpl.append(0)
            win_lose_tie_da_then_maxim_then_wexpl["lose"] += 1
    else:
        if m1 == m2:
            if m1 == "1":
                da_then_maxim_then_wexpl.append(1)
                win_lose_tie_da_then_maxim_then_wexpl["win"] += 1
            else:
                da_then_maxim_then_wexpl.append(0)
                win_lose_tie_da_then_maxim_then_wexpl["lose"] += 1
        else:
            if we1 == "1" and we2 == "1":
                da_then_maxim_then_wexpl.append(1)
                win_lose_tie_da_then_maxim_then_wexpl["win"] += 1
            else:
                da_then_maxim_then_wexpl.append(0)
                if we1=="2" and we2=="2":
                    win_lose_tie_da_then_maxim_then_wexpl["lose"] += 1
                else:
                    win_lose_tie_da_then_maxim_then_wexpl["tie"] += 1

    if m1 == m2:
        if m1 == "1":
            maxim_then_da_then_io.append(1)
        else: 
            maxim_then_da_then_io.append(0)
    else:
        if da1 == da2:
            if da1 == "1":
                maxim_then_da_then_io.append(1)
            else:
                maxim_then_da_then_io.append(0)
        else:
            if io1 == "1" and io2 == "1":
                maxim_then_da_then_io.append(1)
            else:
                maxim_then_da_then_io.append(0)

    if m1 == m2:
        if m1 == "1":
            maxim_then_da_then_wexpl.append(1)
        else: 
            maxim_then_da_then_wexpl.append(0)
    else:
        if da1 == da2:
            if da1 == "1":
                maxim_then_da_then_wexpl.append(1)
            else:
                maxim_then_da_then_wexpl.append(0)
        else:
            if we1 == "1" and we2 == "1":
                maxim_then_da_then_wexpl.append(1)
            else:
                maxim_then_da_then_wexpl.append(0)
print("\nGPT-4o ensembles")
print("DA then maxim accuracy:", len(logic3_acc), 100*np.mean(logic3_acc))
print("Maxim then DA accuracy:", len(logic5_acc), 100*np.mean(logic5_acc))
print("DA + maxim accuracy:", len(logic4_acc), 100*np.mean(logic4_acc))
print("IO then DA accuracy:", len(io_then_da), 100*np.mean(io_then_da))
print("IO then DA then maxim accuracy:", len(io_then_da_then_maxim), 100*np.mean(io_then_da_then_maxim))
print("DA then maxim then IO accuracy:", len(da_then_maxim_then_io), 100*np.mean(da_then_maxim_then_io))
print("DA then maxim then W-Expl accuracy:", len(da_then_maxim_then_wexpl), 100*np.mean(da_then_maxim_then_wexpl))
print("Maxim then DA then IO accuracy:", len(maxim_then_da_then_io), 100*np.mean(maxim_then_da_then_io))
print("Maxim then DA then W-Expl accuracy:", len(maxim_then_da_then_wexpl), 100*np.mean(maxim_then_da_then_wexpl))
op_str.append(round(100*np.mean(logic3_acc),1))
op_str.append(round(100*np.mean(logic5_acc),1))
op_str.append(round(100*np.mean(logic4_acc),1))
# op_str.append(round(100*np.mean(io_then_da),1))
op_str.append(round(100*np.mean(io_then_da_then_maxim),1))
op_str.append(round(100*np.mean(da_then_maxim_then_io),1))
op_str.append(round(100*np.mean(da_then_maxim_then_wexpl),1))
op_str.append(round(100*np.mean(maxim_then_da_then_io),1))
op_str.append(round(100*np.mean(maxim_then_da_then_wexpl),1))

tot = sum(win_lose_tie_da_then_maxim.values())
win_lose_tie_da_then_maxim["win"] = round(100*win_lose_tie_da_then_maxim["win"]/tot,1)
win_lose_tie_da_then_maxim["lose"] = round(100*win_lose_tie_da_then_maxim["lose"]/tot,1)
win_lose_tie_da_then_maxim["tie"] = round(100*win_lose_tie_da_then_maxim["tie"]/tot,1)
op_str_tie.append("DA-then-Maxim: win / tie / loss - " + str(win_lose_tie_da_then_maxim["win"]) + " / " + str(win_lose_tie_da_then_maxim["tie"]) + " / " + str(win_lose_tie_da_then_maxim["lose"]))


tot = sum(win_lose_tie_da_then_maxim_then_wexpl.values())
win_lose_tie_da_then_maxim_then_wexpl["win"] = round(100*win_lose_tie_da_then_maxim_then_wexpl["win"]/tot,1)
win_lose_tie_da_then_maxim_then_wexpl["lose"] = round(100*win_lose_tie_da_then_maxim_then_wexpl["lose"]/tot,1)
win_lose_tie_da_then_maxim_then_wexpl["tie"] = round(100*win_lose_tie_da_then_maxim_then_wexpl["tie"]/tot,1)
op_str_tie.append("DA-then-Maxim-then-W-Expl: win / tie / loss - " + str(win_lose_tie_da_then_maxim_then_wexpl["win"]) + " / " + str(win_lose_tie_da_then_maxim_then_wexpl["tie"]) + " / " + str(win_lose_tie_da_then_maxim_then_wexpl["lose"]))
# op_str_tie.append("DA-then-Maxim-then-W-Expl: " + str(win_lose_tie_da_then_maxim_then_wexpl))


acc_skywork = []
scorediff_skywork = {"pos": [], "neg": []}
for kk in k:
    # if kk not in skywork: continue
    if kk.count("Human:") < condition:
        continue
    if use_valid and kk not in skywork:
        acc_skywork.append(0)
        continue
    if skywork[kk]["score_chosen"] > skywork[kk]["score_rejected"]:
        acc_skywork.append(1)
        scorediff_skywork["pos"].append(skywork[kk]["score_chosen"]-skywork[kk]["score_rejected"])
    else:
        acc_skywork.append(0)
        scorediff_skywork["neg"].append(skywork[kk]["score_rejected"]-skywork[kk]["score_chosen"])
print("\nSkywork-Reward-Llama-3.1-8B-v0.2 accuracy:")
print("Accuracy: ", len(acc_skywork), 100*np.mean(acc_skywork))
print("Confidence:", len(scorediff_skywork["pos"]), np.mean(scorediff_skywork["pos"]),
    len(scorediff_skywork["neg"]), np.mean(scorediff_skywork["neg"]))
op_str.append(round(100*np.mean(acc_skywork),1))

# INF-ORM
acc_inform = []
scorediff_inform = {"pos": [], "neg": []}
for kk in k:
    # if kk not in inform: continue
    if kk.count("Human:") < condition:
        continue
    if use_valid and kk not in inform:
        acc_inform.append(0)
        continue
    if inform[kk]["score_chosen"] > inform[kk]["score_rejected"]:
        acc_inform.append(1)
        scorediff_inform["pos"].append(inform[kk]["score_chosen"]-inform[kk]["score_rejected"])
    else:
        acc_inform.append(0)
        scorediff_inform["neg"].append(inform[kk]["score_rejected"]-inform[kk]["score_chosen"])
print("\nINFORM accuracy:")
print("Accuracy: ", len(acc_inform), 100*np.mean(acc_inform))
print("Confidence:", len(scorediff_inform["pos"]), np.mean(scorediff_inform["pos"]),
    len(scorediff_inform["neg"]), np.mean(scorediff_inform["neg"]))
op_str.append(round(100*np.mean(acc_inform),1))

# QRM
acc_qrm = []
scorediff_qrm = {"pos": [], "neg": []}
for kk in k:
    # if kk not in qrm: continue
    if kk.count("Human:") < condition:
        continue
    if use_valid and kk not in qrm:
        acc_qrm.append(0)
        continue
    if qrm[kk]["score_chosen"] > qrm[kk]["score_rejected"]:
        acc_qrm.append(1)
        scorediff_qrm["pos"].append(qrm[kk]["score_chosen"]-qrm[kk]["score_rejected"])
    else:
        acc_qrm.append(0)
        scorediff_qrm["neg"].append(qrm[kk]["score_rejected"]-qrm[kk]["score_chosen"])
print("\nQRM accuracy:")
print("Accuracy: ", len(acc_qrm), 100*np.mean(acc_qrm))
print("Confidence:", len(scorediff_qrm["pos"]), np.mean(scorediff_qrm["pos"]),
    len(scorediff_qrm["neg"]), np.mean(scorediff_qrm["neg"]))
op_str.append(round(100*np.mean(acc_qrm),1))



# Ensemble w local models
da_then_maxim_then_sky = []
maxim_then_da_then_sky = []
da_then_maxim_then_inform = []
maxim_then_da_then_inform = []
da_then_maxim_then_qrm = []
maxim_then_da_then_qrm = []

# here my logic is that
# if a certain data point does not have a judgment by any one of the judges
# we can still use the output by another judge to judge it, ASSUMING this other judge gives a
# result that is same across the two votes
for kk in k:
    if kk.count("Human:") < condition:
        continue

    # maxim
    if use_valid and kk not in maxim_pairwise_with_result_pred:
        m1 = -1
        m2 = -2
    else:
        if "maxim_pred_1" in maxim_pairwise_with_result_pred[kk]:
            m1 = maxim_pairwise_with_result_pred[kk]["maxim_pred_1"]["Final Answer"]
            m2 = maxim_pairwise_with_result_pred[kk]["maxim_pred_2"]["Final Answer"]
        else:
            if "Final Answer" in maxim_pairwise_with_result_pred[kk]["output_pred_1"]:
                m1 = maxim_pairwise_with_result_pred[kk]["output_pred_1"]["Final Answer"]
                m2 = maxim_pairwise_with_result_pred[kk]["output_pred_2"]["Final Answer"]
            else:
                m1 = maxim_pairwise_with_result_pred[kk]["output_pred_1"]["Answer"]
                m2 = maxim_pairwise_with_result_pred[kk]["output_pred_2"]["Answer"]

    # da
    if use_valid and kk not in da_pred:
        da1 = -1
        da2 = -2 # basically, we have no signal from the da method
    else:
        da1 = da_pred[kk]["output_pred_1"]["Answer"]
        da2 = da_pred[kk]["output_pred_2"]["Answer"]

    # skywork
    if use_valid and kk not in skywork:
        sky1 = "no-score"
        sky2 = "no-score"
    else: 
        sky1 = skywork[kk]["score_chosen"]
        sky2 = skywork[kk]["score_rejected"]

    # inform
    if use_valid and kk not in inform:
        info1 = "no-score"
        info2 = "no-score"
    else: 
        info1 = inform[kk]["score_chosen"]
        info2 = inform[kk]["score_rejected"]

    # qrm
    if use_valid and kk not in qrm:
        q1 = "no-score"
        q2 = "no-score"
    else: 
        q1 = qrm[kk]["score_chosen"]
        q2 = qrm[kk]["score_rejected"]

    if da1 == da2:
        if da1 == "1":
            da_then_maxim_then_sky.append(1)
            da_then_maxim_then_inform.append(1)
            da_then_maxim_then_qrm.append(1)
        else: 
            da_then_maxim_then_sky.append(0)
            da_then_maxim_then_inform.append(0)
            da_then_maxim_then_qrm.append(0)
    else:
        if m1 == m2:
            if m1 == "1":
                da_then_maxim_then_sky.append(1)
                da_then_maxim_then_inform.append(1)
                da_then_maxim_then_qrm.append(1)
            else:
                da_then_maxim_then_sky.append(0)
                da_then_maxim_then_inform.append(0)
                da_then_maxim_then_qrm.append(0)
        else:
            # Skywork
            if sky1 == "no-score":
                da_then_maxim_then_sky.append(0)
            elif sky1 > sky2:
                da_then_maxim_then_sky.append(1)
            else:
                da_then_maxim_then_sky.append(0)
            # INFORM
            if info1 == "no-score":
                da_then_maxim_then_inform.append(0)
            elif info1 > info2:
                da_then_maxim_then_inform.append(1)
            else:
                da_then_maxim_then_inform.append(0)
            # QRM
            if q1 == "no-score":
                da_then_maxim_then_qrm.append(0)
            elif q1 > q2:
                da_then_maxim_then_qrm.append(1)
            else:
                da_then_maxim_then_qrm.append(0)
    
    if m1 == m2:
        if m1 == "1":
            maxim_then_da_then_sky.append(1)
            maxim_then_da_then_inform.append(1)
            maxim_then_da_then_qrm.append(1)
        else: 
            maxim_then_da_then_sky.append(0)
            maxim_then_da_then_inform.append(0)
            maxim_then_da_then_qrm.append(0)

    else:
        if da1 == da2:
            if da1 == "1":
                maxim_then_da_then_sky.append(1)
                maxim_then_da_then_inform.append(1)
                maxim_then_da_then_qrm.append(1)
            else:
                maxim_then_da_then_sky.append(0)
                maxim_then_da_then_inform.append(0)
                maxim_then_da_then_qrm.append(0)
        else:
            # Skywork
            if sky1 == "no-score":
                da_then_maxim_then_sky.append(0)
            elif sky1 > sky2:
                maxim_then_da_then_sky.append(1)
            else:
                maxim_then_da_then_sky.append(0)
            # INFORM
            if info1 == "no-score":
                maxim_then_da_then_inform.append(0)
            elif info1 > info2:
                maxim_then_da_then_inform.append(1)
            else:
                maxim_then_da_then_inform.append(0)                
            # QRM
            if q1 == "no-score":
                maxim_then_da_then_qrm.append(0)
            elif q1 > q2:
                maxim_then_da_then_qrm.append(1)
            else:
                maxim_then_da_then_qrm.append(0)

print("DA then maxim then Skywork accuracy:", len(da_then_maxim_then_sky), 100*np.mean(da_then_maxim_then_sky))
print("Maxim then DA then Skywork accuracy:", len(maxim_then_da_then_sky), 100*np.mean(maxim_then_da_then_sky))
print("DA then maxim then INFORM accuracy:", len(da_then_maxim_then_inform), 100*np.mean(da_then_maxim_then_inform))
print("Maxim then DA then INFORM accuracy:", len(maxim_then_da_then_inform), 100*np.mean(maxim_then_da_then_inform))
print("DA then maxim then QRM accuracy:", len(da_then_maxim_then_qrm), 100*np.mean(da_then_maxim_then_qrm))
print("Maxim then DA then QRM accuracy:", len(maxim_then_da_then_qrm), 100*np.mean(maxim_then_da_then_qrm))

op_str.append(round(100*np.mean(da_then_maxim_then_sky),1))
op_str.append(round(100*np.mean(maxim_then_da_then_sky),1))
op_str.append(round(100*np.mean(da_then_maxim_then_inform),1))
op_str.append(round(100*np.mean(maxim_then_da_then_inform),1))
op_str.append(round(100*np.mean(da_then_maxim_then_qrm),1))
op_str.append(round(100*np.mean(maxim_then_da_then_qrm),1))
'''
print("\n" + dataset + "-" + split)
print("\t".join([str(o) for o in op_str]))

print("\n\n" + "\n".join(op_str_tie))