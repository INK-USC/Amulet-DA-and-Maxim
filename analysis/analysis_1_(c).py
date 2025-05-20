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


DATASET = "Nectar" # Choose from the above list
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

    same_da_conv = 0
    different_da_conv = 0
    wrongly_parsed = 0
    count = 0 
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
            total_turns += len(entries)
            ref_functions = entries[0].functions
            ref_dimensions = entries[0].dimensions
            for conv in entries[1:]:
                if (check_function and conv.functions != ref_functions) or (check_dimension and conv.dimensions != ref_dimensions):
                    different_da_conv += 1
                else:
                    same_da_conv += 1
                ref_functions = conv.functions
                ref_dimensions = conv.dimensions
    print(f'There were {wrongly_parsed} conversations out of {count} which were not parsed successfully')
    print(f'The total turns are: {total_turns}')
    print(same_da_conv, different_da_conv)
    if same_da_conv + different_da_conv + count != total_turns + wrongly_parsed:
        print('ERROR')
    return same_da_conv, different_da_conv


same_da_conv, different_da_conv = count_change(['Human:'])
print(f'The percentage of conversation with different dimension and function across different Human conversation: {different_da_conv / (same_da_conv + different_da_conv):.4f}')
