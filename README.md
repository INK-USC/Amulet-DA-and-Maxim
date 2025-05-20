# Amulet-DA-and-Maxim
# AMULET - (A)nalyze (MUL)ti-(T)urn conversations
This is the official code and associated datasets for the paper titled 

>[AMULET: Putting Complex Multi-Turn Conversations on the Stand with LLM Juries. *Sahana Ramnath, Anurag Mudgil, Brihi Joshi, Skyler Hallinan, Xiang Ren.*]()

For any questions, please contact Sahana Ramnath [sramnath@usc.edu].

## Prompts
We provide all GPT-4o prompts in ``all_prompts.py``.
* ``io_prompt_final``: input - conversation with preference responses, output - preferred response 1 or 2
* ``wexpl_prompt_final``: input - conversation with preference responses, output - preferred response 1 or 2 + NL explanation
* ``dialogact_prompt_final``: input - conversation with preference responses, output - conversation annotated with dialog acts + preferred response 1 or 2 + NL explanation
* ``maxim_prompt_final``: input - conversation with preference responses, output - maxim satisfaction + preferred response 1 or 2 + NL explanation
* ``dialogact_prompt_final_claude`` (for Claude): input - conversation with preference responses, output - conversation annotated with dialog acts in a JSON format + preferred response 1 or 2 + NL explanation

## Loading the dataset
We work with four evaluation datasets: anthropic-train/test, wildfeedback and nectar.
* ``load_all_data.py`` contains the code to load all datasets. 
* ``all_valid_data`` - folder which contains all the cleaned evaluation sets.

## Running the judges
* ``api_[io,wexpl,da,maxim].py`` have the GPT-4o and Claude generation codes. You will have to fill in OpenAI / Claude keys. 
* ``lm_any.py`` has the code to run a local model such as Qwen-2.5-32B-it: ``python lm_any.py --which-dataset anthropic-test --save-path fill-here --which-model Qwen/Qwen2.5-32B-Instruct --which-mode maxim``

The folder ``final_outputs`` has all the DA and maxim annotations we obtained with GPT-4o, Claude and Qwen-2.5-32B-it.

## Running the reward models
The ``reward-models`` folder has the codes to run INF-ORM, QRM, and Skywork-v0.2. 

The folder ``final_outputs`` have all the RM scores we obtained.

## Calculating accuracy of judge and jury
* ``accuracy.py`` gives the accuracy for every judge, RM and jury we have experimented on

## All analyses
Codes and graphs for all analyses in Section 3 in the paper is in the folder ``analysis``.
