import numpy as np
import os
import pickle
import sys
from collections import Counter

# Add file to PYTHON PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evals.extract_dialogue_act import DialogueActResponse

file_paths = {
    "Anthropic Train": "evals/input/da/anthropic-help-train-da-final.pkl",
    "Anthropic Test": "evals/input/da/anthropic-help-test-da-final.pkl",
    "Nectar": "evals/input/da/nectar-train-da-final.pkl",
    "Wild Feedback": "evals/input/da/wildfeedback-train-da-final.pkl"
}

dim_shortname_dict = {
    "Task": "Task",
    "Auto-Feedback": "Auto-Feedback",
    "Allo-Feedback": "Allo-Feedback",
    "Turn Management": "Turn Management",
    "Time Management": "Time Management",
    "Own Communication Management": "OCM",
    "Partner Communication Management": "PCM",
    "Discourse/Interaction Structuring": "DIS",
    "Social Obligations Management": "SOM",
    "Agreement": "Agree",
    "Apology": "Apolg",
    "Disagreement": "Disagree",
    "Initial Self-Introduction": "InitIntro",
    "Answer": "Answer",
    "Feedback": "Fb",
    "Inform": "Inform",
    "Offer": "Offer",
    "Opening": "Opening"
}

func_shortname_dict = {
    "Propositional Question": "PQ",
    "Set Question": "SQ",
    "Choice Question": "CQ",
    "Answer": "Ans",
    "Confirm": "Conf",
    "Disconfirm": "Disconf",
    "Inform": "Info",
    "Agreement": "Agree",
    "Disagreement": "Disagree",
    "Correction": "Corr",
    "Promise": "Prom",
    "Offer": "Offer",
    "Accept Request": "AccReq",
    "Decline Request": "DecReq",
    "Accept Suggest": "AccSug",
    "Decline Suggest": "DecSug",
    "Request": "Req",
    "Instruct": "Instr",
    "Suggest": "Sug",
    
    "Auto-Positive": "Auto+",
    "Auto-Negative": "Auto-",
    
    "Allo-Positive": "Allo+",
    "Allo-Negative": "Allo-",
    "Feedback Elicitation": "FB-Elic",
    
    "Stalling": "Stall",
    "Pausing": "Pause",
    
    "Self-Correction": "SelfCorr",
    "Self-Error": "SelfErr",
    "Retraction": "Retract",
    
    "Completion": "Complete",
    "Correct Misspeaking": "CorrMisspeak",
    
    "Interaction Structuring": "IntStruct",
    "Opening": "Open",
    "Closing": "Close",
    
    "Initial Greeting": "InitGreet",
    "Return Greeting": "RetGreet",
    "Intitial Self-Introduction": "InitIntro",
    "Return Self-Introduction": "RetIntro",
    "Apology": "Apolg",
    "Accept Apology": "AccApolog",
    "Thanking": "Thanks",
    "Accept Thanking": "AccThanks",
    "Initial Goodbye": "InitBye",
    "Return Goodbye": "RetBye",

    "Warn": "Warn",
    "Accept Offer": "AccOffer",
    "Incomplete": "Incompl",
    "Reassurance": "Reassure",
    "Clarification": "Clarify",
    "Clarification Request": "ClarReq",
    "Joking": "Joke",
    "Challenge": "Chall",
    "Disclaim": "Disclaim",
    "Politeness": "Polite",
    "Justify": "Justify",
    "Threat": "Threat",
    "Initial Self-Introduction": "InitIntro",
    "Disrecommend": "Disrecomm",
    "Disagree": "Disagree",
    "Decline Offer": "DecOffer"

}

DATASET = "Wild Feedback"
file_path_da = file_paths[DATASET]
da_pred = pickle.load(open(file_path_da, "rb"))
global total_turns
total_turns = 0 

s = []

def validate_conversation(conversation):
    # Check the order of conversation
    n = len(conversation)
    if n < 2:
        return False
    if not conversation[-2].text[0].startswith("Assistant-1:"):
        return False
    if not conversation[-1].text[0].startswith("Assistant-2:"):
        return False
    for i in range(0, n - 2, 2):
        if not conversation[i].text[0].startswith("Human:"):
            return False
    for i in range(1, n - 2, 2):
        if not conversation[i].text[0].startswith("Assistant:"):
            return False
    return True

def check_if_valid_multi_turn(conversation):
    if not validate_conversation(conversation):
        return False
    # Check the counts of conversation
    human_turns = [conv for conv in conversation if any(s in conv.text[0] for s in ['Human:'])]
    assistant_turns = [conv for conv in conversation if any(s in conv.text[0] for s in ['Assistant:'])]
    chosen_turns = [conv for conv in conversation if any(s in conv.text[0] for s in ['Assistant-1:'])]
    rejected_turns = [conv for conv in conversation if any(s in conv.text[0] for s in ['Assistant-2:'])]
    
    if len(human_turns) == 0 or len(assistant_turns) == 0 or len(chosen_turns) != 1  or len(rejected_turns) != 1:
        return False
    if len(human_turns) - len(assistant_turns) != 1:
        return False
    if len(human_turns) + len(assistant_turns) + len(chosen_turns) + len(rejected_turns) != len(conversation):
        return False
    return True


def count_change(speakers, check_dimension = True, check_function = True):
    
    '''
    Counts how many conversations have consistent or inconsistent dialogue acts (DAs) for the specified speakers.
    Parameters:
    - speakers (list): A list of speaker identifiers (e.g., ['Human', 'Assistant']) whose dialogue acts will be compared.
    - check_dimension (bool): If True, compares the 'dimensions' attribute of dialogue acts across the conversation.
    - check_function (bool): If True, compares the 'functions' attribute of dialogue acts across the conversation.

    Returns:
    - same_da_conv (int): Number of conversations where all specified speakers have the same DA (based on selected checks).
    - different_da_conv (int): Number of conversations where there is a mismatch in DA among the specified speakers.
    '''

    count = 0 
    wrongly_parsed = 0
    functions = {}
    dimensions = {}
    global total_turns
    total_turns = 0
    for conversation, dialogue_act_annotated in da_pred.items():
        response = DialogueActResponse.from_string(dialogue_act_annotated['output_pred_1'])
        count += 1
        if not response.parse_success:
            wrongly_parsed += 1
            continue 
        conversation = response.conversation 
        if not check_if_valid_multi_turn(conversation):
            wrongly_parsed += 1
            continue
        else:
            entries = [conv for conv in conversation if any(s in conv.text[0] for s in speakers)]
            for conv in entries:
                for func in conv.functions:
                    if func in func_shortname_dict:
                        func_name = func_shortname_dict[func]
                    # else:
                    #     func_name = func
                    functions[func_name] = functions.get(func_name, 0) + 1
                for dim in conv.dimensions:
                    if dim in dim_shortname_dict:
                        dim_name = dim_shortname_dict[dim]
                    # else:
                    #     dim_name = dim
                    dimensions[dim_name] = dimensions.get(dim_name, 0) + 1

            total_turns += len(entries)
    print(f'{wrongly_parsed} conversations out of {count} were not parsed successfully')
    print(f'Total turns: {total_turns}')
    # print(functions)
    # print(dimensions)
    functions = {k: round((v*100) / total_turns, 2) for k, v in functions.items()}
    dimensions = {k: round((v*100) / total_turns, 2) for k, v in dimensions.items()}
    return functions, dimensions

def find_top_K(human_values, assistant_values):
    combined_functions_assistant = Counter(assistant_values)
    top_10_functions_assistant = dict(combined_functions_assistant.most_common(10))
    combined_functions_human = Counter(human_values)
    top_10_functions_human = dict(combined_functions_human.most_common(5))
    top_10_functions = list(top_10_functions_human.keys())
    for func in top_10_functions_assistant:
        if func in top_10_functions:
            continue 
        top_10_functions.append(func)
        if len(top_10_functions) == 10:
            break

    human_result, assistant_result = {}, {}
    for func in top_10_functions:
        human_result[func] = human_values.get(func, 0)
        assistant_result[func] = assistant_values.get(func, 0)
    return human_result, assistant_result

functions_human, dimensions_human = count_change(['Human:'])
functions_assistant, dimensions_assistant = count_change(['Assistant:'])

human_result, assistant_result = find_top_K(functions_human, functions_assistant)
print(human_result)
print(assistant_result)
human_result, assistant_result = find_top_K(dimensions_human, dimensions_assistant)
print(human_result)
print(assistant_result)