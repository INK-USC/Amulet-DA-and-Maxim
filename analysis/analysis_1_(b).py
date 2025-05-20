import numpy as np
import os
import pickle
import sys

# Add file to PYTHON PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evals.extract_dialogue_act import DialogueActResponse

file_paths = {
    "Anthropic Train": "evals/input/da/anthropic-help-train-da-final.pkl",
    "Anthropic Test": "evals/input/da/anthropic-help-test-da-final.pkl",
    "Nectar": "evals/input/da/nectar-train-da-final.pkl",
    "Wild Feedback": "evals/input/da/wildfeedback-train-da-final.pkl"
}


DATASET = "Wild Feedback" # Choose from the above list
file_path_da = file_paths[DATASET]
da_pred = pickle.load(open(file_path_da, "rb"))

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
    number_of_different_da = {}
    total_turns = 0
    for orig_conversation, dialogue_act_annotated in da_pred.items():
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
            da = {}
            for conv in entries:
                current_functions_and_dimension = (tuple(conv.functions), tuple(conv.dimensions))  
                da[current_functions_and_dimension] = da.get(current_functions_and_dimension, 0) + 1
            number_of_different_da[len(da)] =  number_of_different_da.get(len(da), 0) + 1
            total_turns += len(entries)

    print(f'{wrongly_parsed} conversations out of {count} were not parsed successfully')
    print(total_turns)
    number_of_different_da = dict(sorted(number_of_different_da.items()))
    return number_of_different_da

number_of_different_da = count_change(['Human:'])
print(number_of_different_da)
s = 0 
for v in number_of_different_da.values():
    s += v
print(s)