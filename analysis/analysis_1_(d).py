## Percentage of times assistant turn changes when human turns changes


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


DATASET = "Anthropic Train" # Choose from the above list
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

def count_change():
    
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
    total_turns = 0
    correct_change = 0 
    wrong_change = 0 

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
            entries_human = [conv for conv in conversation if any(s in conv.text[0] for s in ['Human:'])]
            entries_human = entries_human[:-1]
            entries_assistant = [conv for conv in conversation if any(s in conv.text[0] for s in ['Assistant:'])]
            if len(entries_human) != len(entries_assistant):
                print('Error!!')
            prev_human = None
            prev_assistant = None
            for conv_human, conv_assistant in zip(entries_human, entries_assistant) :
                curr_human = tuple(conv_human.functions)
                curr_assistant = tuple(conv_assistant.functions)
                if prev_human is not None and prev_human != curr_human:
                    if prev_assistant == curr_assistant:
                        wrong_change += 1
                    else:
                        correct_change += 1
                prev_human = curr_human
                prev_assistant = curr_assistant
            total_turns += (len(entries_human) + len(entries_assistant))

    print(f'{wrongly_parsed} conversations out of {count} were not parsed successfully')
    print(f'Total turns: {total_turns}')
    return correct_change, wrong_change


correct_change, wrong_change = count_change()
print(correct_change, wrong_change)
print(correct_change/(correct_change + wrong_change))