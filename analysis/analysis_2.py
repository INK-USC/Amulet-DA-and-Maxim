import os
import pickle
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
from evals.extract_maxims import MaximResponse


file_paths_maxims = {
    "Anthropic Train": "evals/input/maxim/anthropic-help-train-maxim-final.pkl",
    "Anthropic Test": "evals/input/maxim/anthropic-help-test-maxim-final.pkl",
    "Nectar": "evals/input/maxim/nectar-train-maxim-final.pkl",
    "Wild Feedback": "evals/input/maxim/wildfeedback-train-maxim-final.pkl"
}

DATASET = "Wild Feedback"
file_path = file_paths_maxims[DATASET] # Choose dataset from above list
maxims_pred = pickle.load(open(file_path, "rb"))


def get_maxims():

    maxims = []
    for conversation, maxim_annotated in maxims_pred.items():
        if 'maxim_pred_1' in maxim_annotated:
            response = MaximResponse.from_string(maxim_annotated['maxim_pred_1'])
        else:
            response = MaximResponse.from_string(maxim_annotated['output_pred_1'])
        maxims.append(response.maxims)
    return maxims
        
maxims = get_maxims()
maxim_count = defaultdict(lambda: [0, 0, 0, 0])

for maxim_dict in maxims:
    for key, value in maxim_dict.items():
        if value == '1':
            maxim_count[key][0] += 1
        elif value == '2':
            maxim_count[key][1] += 1
        elif value == 'both':
            maxim_count[key][2] += 1
        else:
            maxim_count[key][3] += 1

print(maxim_count)
normalised_count = {}
for key, values in maxim_count.items():
    total = sum(values)
    normalized = [round(v / total, 4) for v in values]
    normalised_count[key] = normalized

print(normalised_count)

