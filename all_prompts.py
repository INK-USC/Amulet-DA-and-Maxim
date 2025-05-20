io_prompt_final = """Instruction: You will be given a dialog conversation between a human user and an LLM assistant. The dialog is split into turns - a turn is defined as an utterance by either the human user or the assistant. Note that the roles of "speaker" (S) and "addressee" (A) will alternate at every turn. The last turn alone will have two responses, sampled from different LLM assistants. Your task is to analyze the two responses and say which one of them is better - you should take all the previous turns of the dialog into consideration. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. An example is given below with the required JSON output format. Say "1" if you think Assistant-1's response is better and "2" if you think Assistant-2's response is better.

Example Dialog - 
Human: Human's turn
Assistant: Assistant's turn
Human: Human's turn
Assistant: Assistant's turn
Human: Human's turn
Assistant-1: Assistant's turn by Assistant-1
Assistant-2: Assistant's turn by Assistant-2

Example Output - 
{
	"Answer": "fill either 1 or 2"
}
"""

wexpl_prompt_final = """Instruction: You will be given a dialog conversation between a human user and an LLM assistant. The dialog is split into turns - a turn is defined as an utterance by either the human user or the assistant. Note that the roles of "speaker" (S) and "addressee" (A) will alternate at every turn. The last turn alone will have two responses, sampled from different LLM assistants. Your task is to analyze the two responses and say which one of them is better - you should take all the previous turns of the dialog into consideration. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. An example is given below with the required JSON output format. Say "1" if you think Assistant-1's response is better and "2" if you think Assistant-2's response is better.

Example Dialog - 
Human: Human's turn
Assistant: Assistant's turn
Human: Human's turn
Assistant: Assistant's turn
Human: Human's turn
Assistant-1: Assistant's turn by Assistant-1
Assistant-2: Assistant's turn by Assistant-2

Example Output - 
{
	"Answer": "fill either 1 or 2",
	"Explanation": "fill your explanation here"
}
"""

# print(baseline_prompt_final)

dialogact_prompt_final = """Instruction: You will be given a dialog conversation between a human user and an LLM assistant. The dialog is split into turns - a turn is defined as an utterance by either the human user or the assistant. Note that the roles of "speaker" (S) and "addressee" (A) will alternate at every turn. The last turn alone will have two responses, sampled from different LLM assistants. Your task is to label each turn of dialogue in terms of dialog-acts - dialog acts are defined in terms of communicative dimensions "Dim" and corresponding communicative functions "Func" [detailed below along with their meanings]. For each turn of dialogue, you must mark *all* the dimensions and functions that are present. You should take the previous turns of the dialog into consideration when labeling the dialog acts. Finally, use these dialog acts to determine which response is better - say "1" if you think Assistant-1's response is better and "2" if you think Assistant-2's response is better. Also provide an explanation for your choice. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. Examples are given below with the desired output format.

List of dimensions and functions -  
(1) Dim: Task, Meaning: Underlying task/activity, Func: Propositional Question, Set Question, Choice Question, Answer, Confirm, Disconfirm, Inform, Agreement, Disagreement, Correction, Promise, Offer, Accept Request, Decline Request, Accept Suggest, Decline Suggest, Request, Instruct, Suggest
(2) Dim: Auto-Feedback, Meaning: Speaker's processing of previous utterances, Func: Auto-Positive, Auto-Negative
(3) Dim: Allo-Feedback, Meaning: Speaker expressing opinions about addressee's processing of previous utterances, Func: Allo-Positive, Allo-Negative, Feedback Elicitation
(4) Dim: Time Management, Meaning: Concerning the allocation of time to the speaker, Func: Stalling, Pausing
(6) Dim: Own Communication Management, Meaning: Editing speaker's own speech within the current turn, Func: Self-Correction, Self-Error, Retraction
(7) Dim: Partner Communication Management, Meaning: Editing what the addressee said, Func: Completion, Correct Misspeaking, 
(8) Dim: Discourse/Interaction Structuring, Meaning: Explicitly structuring the interaction, Func: Interaction Structuring, Opening, Closing
(9) Dim: Social Obligations Management, Meaning: Social obligation, Func: Initial Greeting, Return Greeting, Intitial Self-Introduction, Return Self-Introduction, Apology, Accept Apology, Thanking, Accept Thanking, Initial Goodbye, Return Goodbye

You should use only the above dimensions and functions, do not make up new ones. Below you will find detailed definitions of the functions. To reiterate, S refers to the speaker and A refers to the addressee. Both the human and the LM assistant take on the roles of S and A in alternating turns.

Information-seeking functions - 
* Propositional Question: Function performed by S, in order to know whether the proposition, which forms the semantic content, is true. S assumes that A knows whether the proposition is true or not, and puts pressure on A to provide this information.
* Set Question: Function performed by S, in order to know which elements of a given set have a certain property specified by the semantic content; S puts pressure on A to provide this information, which S assumes that A possesses. S believes that at least one element of the set has that property.
* Choice Question: Function performed by S, in order to know which one from a list of alternative propositions, specified by the semantic content, is true; S believes that exactly one element of that list is true; S assumes that A knows which of the alternative propositions is true, and S puts pressure on A to provide this information.

 Information-providing functions - 
 * Answer: Function performed by S, in order to make certain information available to A which S believes A wants to know; S assumes that this information is correct.
* Confirm: Function performed by S, in order to inform A that certain information that A wants to know, and concerning which A holds an uncertain belief, is indeed correct.
* Disconfirm: Function performed by S, in order to let A know that certain information that A wants to know, and concerning which A holds an uncertain belief, is incorrect.
* Inform: Function performed by S, in order to make the information contained in the semantic content known to A; S assumes that the information is correct.
* Agreement: Function performed by S, in order to inform A that S assumes a given proposition to be true, which S believes that A also assumes to be true.
* Disagreement: Function performed by S, in order to inform A that S assumes a given proposition to be false, which S believes that A assumes to be true.
* Correction: Function performed by S, in order to inform A that certain information which S has reason to believe that A assumes to be correct, is in fact incorrect and that instead the information that S provides is correct.

Commissive functions - 
* Promise: Function by which S commits to perform the action specified in the semantic content, in the manner or with the frequency or depending on the conditions that S makes explicit. S believes that this action would be in A's interest.
* Offer: Function by which S indicates willingness and ability to perform the action specified by the semantic content, conditional on the consent of A that S do so.
* Accept Request: Function by which S commits to perform an action that S has been requested to perform, possibly depending on certain conditions that S makes explicit.
* Decline Request: Function by which S refuses to perform an action that S has been requested to perform, possibly depending on certain conditions that S makes explicit.
* Accept Suggest: Function by which S commits to perform an action that was suggested, possibly with certain restrictions or conditions concerning manner or frequency of performance.
* Decline Suggest: Function by which S indicates that S will not perform an action that was suggested, possibly depending on certain conditions that S makes explicit.

Directive functions - 
* Request: Function performed by S, in order to create a commitment for A to perform a certain action in the manner or with the frequency described by the semantic content, conditional on A's consent to perform the action. S assumes that A is able to perform this action.
* Instruct: Function performed by S, in order to create a commitment for A to carry out a named action in the manner or with the frequency specified by the semantic content; S assumes that A is able and willing to carry out the action.
* Suggest: Function performed by S, in order to make A consider the performance of a certain action specified by the semantic content. S believes that this action is in A's interest, and assumes that A is able to perform the action.

Feedback functions - 
* Auto-Positive: Function performed by S, in order to inform A that S believes that S's processing of the previous utterance(s) was successful.
* Allo-Positive: Function performed by S, in order to inform A that S believes that A's processing of the previous utterance(s) was successful.
* Auto-Negative: Function performed by S, in order to inform A that S's processing of the previous utterance(s) encountered a problem.
* Allo-Negative: Function performed by S, in order to inform A that S believes that A's processing of the previous utterance(s) encountered a problem.
* Feedback Elicitation: Function performed by S, in order to know whether A's processing of the previous utterance(s) was successful.

Time management functions - 
* Stalling: Function performed by S, in order to have a little extra time to construct S's contribution.
* Pausing: Function performed by S, in order to suspend the dialogue for a short while.

Own and Partner Communication Management Functions -
* Completion: Function performed by S in order to assist A in the completion of an utterance.
* Correct Misspeaking: Function performed by S, in order to correct (part of) an utterance by A assuming that A made a speaking error.
* Self-Error: Function performed by S, in order to signal to the A that S has made a mistake in speaking.
* Retraction: Function performed by S, in order to withdraw something that S just said within the same turn.
* Self-Correction: Function performed by S, in order to correct an error that S just made, or to improve on an infelicitous formulation that S just used, within the same turn.

Discourse structuring functions - 
* Interaction Structuring: Function performed by S, in order to explicitly indicate to A the function or topic of S's next contribution(s).
* Opening: Function performed by S, in order to inform A that S is ready and willing to engage in a dialogue with A.
* Closing: Function performed by S, in order to inform A that S is about to end the conversation.

 Social obligations management functions - 
 * Initial Greeting: Function performed by S, in order to inform A that S is present and aware of A's presence; S puts pressure on A to acknowledge this.
* Return Greeting: Function performed by S, in order to acknowledge that S is aware of A's presence, and of A having signalled A's presence to S.
* Initial Self-Introduction: Function performed by S, in order to make A know about S; S puts pressure on A to acknowledge this.
* Return Self-Introduction: Function performed by S, in order to make A know about S in response to a self-introduction by A.
* Apology: Function performed by S, that signals S wants A to knows that S regrets something; S puts pressure on A to acknowledge this.
* Accept Apology: Function performed by S, in order to mitigate the feelings of regret that A has expressed.
* Thanking: Function performed by S, in order to inform A that S is grateful for some action performed by A; S puts pressure on A to acknowledge this.
* Accept Thanking: Function performed by S, in order to mitigate the feelings of gratitude which A has expressed.
* Initial Goodbye: Function performed by S, in order to inform A that S intends the current utterance to be S's final contribution to the dialogue; S puts pressure on A to acknowledge this.
* Return Goodbye: Function performed by S, in order to acknowledge awareness that A has signalled A's final contribution to the dialogue and S signals in return S's agreement to end the dialogue; S has been pressured to respond to an initial goodbye by A.

Important additional note: If a turn contains the answer to a question from a prior turn, make sure to mark the communicative function "Answer" in addition to any other function present in the turn (such as "Inform", "Suggest", "Instruct", etc.). That is, even if the answer is phrased as an instruction, or a suggestion, etc., if the turn includes the answer to a question from a prior turn, specify "Answer" as one of the functions in the turn.

Example Dialog 1 - 
Human: How do I teach kids to meditate?
Assistant: Great question! That's a really useful skill to cultivate, it can bring peace, calm, and happiness. I'm glad you want to teach your kids about it.
Human: All right, so how do we start?
Assistant: Well, we can get started with just being silent. You can tell the kids it's okay if they just sit there quietly for a few minutes without thinking of anything.
Human: any other ideas? they are fidgeting
Assistant-1: Kids are great at fidgeting, so let's make sure we give them permission to fidget.
Assistant-2: Yeah, definitely. What works for you personally? What works for other people?

Example Output 1 - 
Human: How do I teach kids to meditate? <SEP> {"Dim": "Task", "Func": "Set Question"}
Assistant: Great question! That's a really useful skill to cultivate, it can bring peace, calm, and happiness. I'm glad you want to teach your kids about it. <SEP> {"Dim": "Allo-Feedback", "Func": "Allo-Positive"} {"Dim": "Task", "Func": "Inform"}  
Human: All right, so how do we start? <SEP> {"Dim": "Allo-Feedback", "Func": "Allo-Positive"} {"Dim": "Task", "Func": "Request & Set Question"}
Assistant: Well, we can get started with just being silent. You can tell the kids it's okay if they just sit there quietly for a few minutes without thinking of anything. <SEP> {"Dim": "Time Management", "Func": "Stalling"} {"Dim": "Task", "Func": "Answer"}
Human: any other ideas? they are fidgeting <SEP> {"Dim": "Allo-Feedback", "Func": "Allo-Negative"} {"Dim": "Task", "Func": "Set Question & Inform"}
Assistant-1: Kids are great at fidgeting, so let's make sure we give them permission to fidget. <SEP> {"Dim": "Task", "Func": "Inform & Instruct"}
Assistant-2: Yeah, definitely. What works for you personally? What works for other people? <SEP> {"Dim": "Auto-Feedback", "Func": "Auto-Positive"} {"Dim": "Task", "Func": "Set Question"}

{
	"Answer": "fill either 1 or 2",
	"Explanation": "fill your explanation here"
}

Example Dialog 2 - 
Human: what are some good novels for a 9 year old?
Assistant: That depends on the 9 year old, but if they like thrillers or mysteries then maybe something like "The Body in the Bog" or "The Boy Who Followed Ripley." For more modern, female-written fiction I might recommend "Jane Eyre" or "Frankenstein."
Human: That sounds cool
Assistant-1: I'm a programmer, so I'm really into good programming texts, like "The Practice of Programming" or "JavaScript: The Good Parts." But that might be a bit too much for a 9 year old, so
Assistant-2: Do you want some quotes?

Example Output 2 - 
Human: what are some good novels for a 9 year old? <SEP> {"Dim": "Task", "Func": "Set Question"}
Assistant: That depends on the 9 year old, but if they like thrillers or mysteries then maybe something like "The Body in the Bog" or "The Boy Who Followed Ripley." For more modern, female-written fiction I might recommend "Jane Eyre" or "Frankenstein." <SEP> {"Dim": "Answer & Suggest"}
Human: That sounds cool <SEP> {"Dim": "Allo-Feedback", "Func": "Allo-Positive"}
Assistant-1: I'm a programmer, so I'm really into good programming texts, like "The Practice of Programming" or "JavaScript: The Good Parts." But that might be a bit too much for a 9 year old, so <SEP> {"Dim": "Task", "Func": "Inform"} {"Dim": "Own Communication Management", "Func": "Retraction"}
Assistant-2: Do you want some quotes? <SEP> {"Dim": "Task", "Func": "Offer & Propositional Question"}

{
	"Answer": "fill either 1 or 2",
	"Explanation": "fill your explanation here"
}

Dialog - 
"""

maxim_prompt_final = """Instruction: You will be given a dialog conversation between a human user and an LLM assistant. The dialog is split into turns - a turn is defined as an utterance by either the human user or the assistant. Note that the roles of "speaker" and "addressee" will alternate at every turn. The last turn alone will have two responses, sampled from different LLM assistants.  Your task is to analyze the two responses and say which one of them is better using the below maxim based evaluation protocol. 
For each of the twelve maxims described below, indicate whether response 1 (from Assistant-1) or response 2 (from Assistant-2) is better in satisfying the maxim by specifying "1" or "2" respectively. If both responses satisfy the maxim equally, say "both" and if neither response satisfies the maxim, say "neither". Finally provide a free-text explanation for your decisions and provide your answer of whether response 1 or response 2 is better. You should take the previous turns of the dialog into consideration when labeling the satisfaction of maxims and when providing your final answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. An example is given below with the required JSON output format. Say "1" if you think Assistant-1's response is better and "2" if you think Assistant-2's response is better. You have to pick "1" or "2", you cannot say "both" for the final answer.

List of Grice's maxims -  
(1) Quantity-1: The response should provide a sufficient amount of information.
(2) Quantity-2: The response should not contain unnecessary details.
(3) Quality: The response should be factual and supported by adequate evidence whenever possible.
(4) Relevance-1: The response should directly and relevantly address the recipient's statements in a helpful manner.
(5) Relevance-2:  The response should be relevant to the current topic and not unnaturally shift the conversation to unrelated subjects.
(6) Manner-1: The response should be clear, unambiguous, and presented in a well-organized fashion.
(7) Manner-2: The response should be accessible and use appropriate language tailored to the recipient's level of understanding.
(8) Benevolence-1: The response should not exhibit insensitivity, rudeness, or harm.
(9) Benevolence-2: The response should not reflect an engagement or endorsement with requests that are harmful or unethical.
(10) Transparency-1: The response should recognize the speaker's knowledge boundaries, making clear any limitations in expertise, evidence, experience, or context.
(11) Transparency-2: The response should recognize the speaker's operational capabilities, highlighting the nature of actions that can or cannot be performed.
(12) Transparency-3: The response should be forthright about the speaker's willingness to engage with specific subjects or heed relevant advice.

You should use only the above maxims, do not make up new ones. 

Example Dialog 1 - 
Human: What do I do about rough knees?
Assistant: Oh, is there something I can help with?
Human: I want my knees to feel smoother
Assistant: You want to have smoother knees?
Human: Yes
Assistant-1: So you want your knees to not be rough?
Assistant-2: I'm happy to help with that, but first I'm going to ask you some questions to help me understand how you feel, and you can tell me anything that's on your mind. I'll start by asking you to tell me about the thing that's making you the most frustrated, or the thing that's bothering you the most. What's making you the most frustrated? What is bothering you the most

Example Output - 1
{
	"Quantity-1": "neither",
	"Quantity-2": "1",
	"Quality": "1",
	"Relevance-1": "neither",
	"Relevance-2": "1",
	"Manner-1": "1",
	"Manner-2": "both",
	"Benevolence-1": "both",
	"Benevolence-2": "both",
	"Transparency-1": "both",
	"Transparency-2": "both",
	"Transparency-3": "both",
	"Explanation": "fill your explanation here",
	"Final Answer": "fill either 1 or 2"
}

Example Dialog 2 -
Human: Is a higher octane gasoline better for your car?
Assistant: How old is the car? What is the mileage? What do you typically put in it? What model car is it?
Human: a 10 year old Ford.
Assistant: Then you probably don't need high-octane gasoline. The average car will do fine with the regular fuel.
Human: Then why do they sell it?
Assistant-1: Automotive manufacturers sell the high-octane fuel for you to use to make the engine run better. It's a trade-off. The fuel has more energy than the regular gas, but it also burns faster and can make your engine more prone to damage.
Assistant-2: There are various reasons. The gasoline may not be entirely identical, and if you have a few items that are very sensitive to fuel, you may need it for your car. However, most items should be OK with regular gas, and probably wouldn't be damaged by it either.

Example Output - 2
{
	"Quantity-1": "1",
	"Quantity-2": "1",
	"Quality": "1",
	"Relevance-1": "1",
	"Relevance-2": "1",
	"Manner-1": "1",
	"Manner-2": "1",
	"Benevolence-1": "both",
	"Benevolence-2": "both",
	"Transparency-1": "1",
	"Transparency-2": "both",
	"Transparency-3": "both"
	"Explanation": "fill your explanation here",
	"Final Answer": "fill either 1 or 2"
}

Example Dialog 3 -
Human: Can you tell me the steps for getting a harbor boat slip for my sailboat?
Assistant: Sure. I believe you'll need a copy of the marina rules and regulations, since some of those details will likely determine what forms you'll need to fill out. It looks like there's a quick way to get a copy of the rules by phone, since you might be able to get the contact information for the harbormaster from your yacht club. If you get a hold of that person, you'll also be
Human: Thanks. And what are some typical forms required?
Assistant-1: Oh, I'm afraid I'm not very familiar with these processes myself. If you want, I can look it up for you, or you can ask me to do so when you need the information.
Assistant-2: At the marina level, there's usually the rental contract that covers the slip, and the paperwork to get insurance for the boat. You'll also need to register your boat with the state, usually by mail, but they'll also likely be able to help you if you go to the state office in person. At the yacht club level, you'll usually need

Example Output - 3
{
	"Quantity-1": "2",
	"Quantity-2": "2",
	"Quality": "2",
	"Relevance-1": "2",
	"Relevance-2": "both",
	"Manner-1": "2",
	"Manner-2": "both",
	"Benevolence-1": "both",
	"Benevolence-2": "both",
	"Transparency-1": "1",
	"Transparency-2": "both",
	"Transparency-3": "both"
	"Explanation": "fill your explanation here",
	"Final Answer": "fill either 1 or 2"
}

Dialog - 
"""

dialogact_prompt_final_claude = """Instruction: You will be given a dialog conversation between a human user and an LLM assistant. The dialog is split into turns - a turn is defined as an utterance by either the human user or the assistant. Note that the roles of "speaker" (S) and "addressee" (A) will alternate at every turn. The last turn alone will have two responses, sampled from different LLM assistants. Your task is to label each turn of dialogue in terms of dialog-acts - dialog acts are defined in terms of communicative dimensions "Dim" and corresponding communicative functions "Func" [detailed below along with their meanings]. For each turn of dialogue, you must mark *all* the dimensions and functions that are present. You should take the previous turns of the dialog into consideration when labeling the dialog acts. Finally, use these dialog acts to determine which response is better - say "1" if you think Assistant-1's response is better and "2" if you think Assistant-2's response is better. Also provide an explanation for your choice. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. Examples are given below with the desired output format. Strictly follow the provided output format, do not generate anything more than what is needed.

List of dimensions and functions -  
(1) Dim: Task, Meaning: Underlying task/activity, Func: Propositional Question, Set Question, Choice Question, Answer, Confirm, Disconfirm, Inform, Agreement, Disagreement, Correction, Promise, Offer, Accept Request, Decline Request, Accept Suggest, Decline Suggest, Request, Instruct, Suggest
(2) Dim: Auto-Feedback, Meaning: Speaker's processing of previous utterances, Func: Auto-Positive, Auto-Negative
(3) Dim: Allo-Feedback, Meaning: Speaker expressing opinions about addressee's processing of previous utterances, Func: Allo-Positive, Allo-Negative, Feedback Elicitation
(4) Dim: Time Management, Meaning: Concerning the allocation of time to the speaker, Func: Stalling, Pausing
(6) Dim: Own Communication Management, Meaning: Editing speaker's own speech within the current turn, Func: Self-Correction, Self-Error, Retraction
(7) Dim: Partner Communication Management, Meaning: Editing what the addressee said, Func: Completion, Correct Misspeaking, 
(8) Dim: Discourse/Interaction Structuring, Meaning: Explicitly structuring the interaction, Func: Interaction Structuring, Opening, Closing
(9) Dim: Social Obligations Management, Meaning: Social obligation, Func: Initial Greeting, Return Greeting, Intitial Self-Introduction, Return Self-Introduction, Apology, Accept Apology, Thanking, Accept Thanking, Initial Goodbye, Return Goodbye

You should use only the above dimensions and functions, do not make up new ones. Below you will find detailed definitions of the functions. To reiterate, S refers to the speaker and A refers to the addressee. Both the human and the LM assistant take on the roles of S and A in alternating turns.

Information-seeking functions - 
* Propositional Question: Function performed by S, in order to know whether the proposition, which forms the semantic content, is true. S assumes that A knows whether the proposition is true or not, and puts pressure on A to provide this information.
* Set Question: Function performed by S, in order to know which elements of a given set have a certain property specified by the semantic content; S puts pressure on A to provide this information, which S assumes that A possesses. S believes that at least one element of the set has that property.
* Choice Question: Function performed by S, in order to know which one from a list of alternative propositions, specified by the semantic content, is true; S believes that exactly one element of that list is true; S assumes that A knows which of the alternative propositions is true, and S puts pressure on A to provide this information.

 Information-providing functions - 
 * Answer: Function performed by S, in order to make certain information available to A which S believes A wants to know; S assumes that this information is correct.
* Confirm: Function performed by S, in order to inform A that certain information that A wants to know, and concerning which A holds an uncertain belief, is indeed correct.
* Disconfirm: Function performed by S, in order to let A know that certain information that A wants to know, and concerning which A holds an uncertain belief, is incorrect.
* Inform: Function performed by S, in order to make the information contained in the semantic content known to A; S assumes that the information is correct.
* Agreement: Function performed by S, in order to inform A that S assumes a given proposition to be true, which S believes that A also assumes to be true.
* Disagreement: Function performed by S, in order to inform A that S assumes a given proposition to be false, which S believes that A assumes to be true.
* Correction: Function performed by S, in order to inform A that certain information which S has reason to believe that A assumes to be correct, is in fact incorrect and that instead the information that S provides is correct.

Commissive functions - 
* Promise: Function by which S commits to perform the action specified in the semantic content, in the manner or with the frequency or depending on the conditions that S makes explicit. S believes that this action would be in A's interest.
* Offer: Function by which S indicates willingness and ability to perform the action specified by the semantic content, conditional on the consent of A that S do so.
* Accept Request: Function by which S commits to perform an action that S has been requested to perform, possibly depending on certain conditions that S makes explicit.
* Decline Request: Function by which S refuses to perform an action that S has been requested to perform, possibly depending on certain conditions that S makes explicit.
* Accept Suggest: Function by which S commits to perform an action that was suggested, possibly with certain restrictions or conditions concerning manner or frequency of performance.
* Decline Suggest: Function by which S indicates that S will not perform an action that was suggested, possibly depending on certain conditions that S makes explicit.

Directive functions - 
* Request: Function performed by S, in order to create a commitment for A to perform a certain action in the manner or with the frequency described by the semantic content, conditional on A's consent to perform the action. S assumes that A is able to perform this action.
* Instruct: Function performed by S, in order to create a commitment for A to carry out a named action in the manner or with the frequency specified by the semantic content; S assumes that A is able and willing to carry out the action.
* Suggest: Function performed by S, in order to make A consider the performance of a certain action specified by the semantic content. S believes that this action is in A's interest, and assumes that A is able to perform the action.

Feedback functions - 
* Auto-Positive: Function performed by S, in order to inform A that S believes that S's processing of the previous utterance(s) was successful.
* Allo-Positive: Function performed by S, in order to inform A that S believes that A's processing of the previous utterance(s) was successful.
* Auto-Negative: Function performed by S, in order to inform A that S's processing of the previous utterance(s) encountered a problem.
* Allo-Negative: Function performed by S, in order to inform A that S believes that A's processing of the previous utterance(s) encountered a problem.
* Feedback Elicitation: Function performed by S, in order to know whether A's processing of the previous utterance(s) was successful.

Time management functions - 
* Stalling: Function performed by S, in order to have a little extra time to construct S's contribution.
* Pausing: Function performed by S, in order to suspend the dialogue for a short while.

Own and Partner Communication Management Functions -
* Completion: Function performed by S in order to assist A in the completion of an utterance.
* Correct Misspeaking: Function performed by S, in order to correct (part of) an utterance by A assuming that A made a speaking error.
* Self-Error: Function performed by S, in order to signal to the A that S has made a mistake in speaking.
* Retraction: Function performed by S, in order to withdraw something that S just said within the same turn.
* Self-Correction: Function performed by S, in order to correct an error that S just made, or to improve on an infelicitous formulation that S just used, within the same turn.

Discourse structuring functions - 
* Interaction Structuring: Function performed by S, in order to explicitly indicate to A the function or topic of S's next contribution(s).
* Opening: Function performed by S, in order to inform A that S is ready and willing to engage in a dialogue with A.
* Closing: Function performed by S, in order to inform A that S is about to end the conversation.

 Social obligations management functions - 
 * Initial Greeting: Function performed by S, in order to inform A that S is present and aware of A's presence; S puts pressure on A to acknowledge this.
* Return Greeting: Function performed by S, in order to acknowledge that S is aware of A's presence, and of A having signalled A's presence to S.
* Initial Self-Introduction: Function performed by S, in order to make A know about S; S puts pressure on A to acknowledge this.
* Return Self-Introduction: Function performed by S, in order to make A know about S in response to a self-introduction by A.
* Apology: Function performed by S, that signals S wants A to knows that S regrets something; S puts pressure on A to acknowledge this.
* Accept Apology: Function performed by S, in order to mitigate the feelings of regret that A has expressed.
* Thanking: Function performed by S, in order to inform A that S is grateful for some action performed by A; S puts pressure on A to acknowledge this.
* Accept Thanking: Function performed by S, in order to mitigate the feelings of gratitude which A has expressed.
* Initial Goodbye: Function performed by S, in order to inform A that S intends the current utterance to be S's final contribution to the dialogue; S puts pressure on A to acknowledge this.
* Return Goodbye: Function performed by S, in order to acknowledge awareness that A has signalled A's final contribution to the dialogue and S signals in return S's agreement to end the dialogue; S has been pressured to respond to an initial goodbye by A.

Important additional note: If a turn contains the answer to a question from a prior turn, make sure to mark the communicative function "Answer" in addition to any other function present in the turn (such as "Inform", "Suggest", "Instruct", etc.). That is, even if the answer is phrased as an instruction, or a suggestion, etc., if the turn includes the answer to a question from a prior turn, specify "Answer" as one of the functions in the turn.

Example Dialog 1 - 
Human: How do I teach kids to meditate?
Assistant: Great question! That's a really useful skill to cultivate, it can bring peace, calm, and happiness. I'm glad you want to teach your kids about it.
Human: All right, so how do we start?
Assistant: Well, we can get started with just being silent. You can tell the kids it's okay if they just sit there quietly for a few minutes without thinking of anything.
Human: any other ideas? they are fidgeting
Assistant-1: Kids are great at fidgeting, so let's make sure we give them permission to fidget.
Assistant-2: Yeah, definitely. What works for you personally? What works for other people?

Example Output 1 -
{
'''Human: How do I teach kids to meditate?''': '''{"Dim": "Task", "Func": "Set Question"}''',
'''Assistant: Great question! That's a really useful skill to cultivate, it can bring peace, calm, and happiness. I'm glad you want to teach your kids about it.''': '''{"Dim": "Allo-Feedback", "Func": "Allo-Positive"} {"Dim": "Task", "Func": "Inform"}''',
'''Human: All right, so how do we start?': '{"Dim": "Allo-Feedback", "Func": "Allo-Positive"} {"Dim": "Task", "Func": "Request & Set Question"}''',
'''Assistant: Well, we can get started with just being silent. You can tell the kids it's okay if they just sit there quietly for a few minutes without thinking of anything.''': '''{"Dim": "Time Management", "Func": "Stalling"} {"Dim": "Task", "Func": "Answer"}''',
'''Human: any other ideas? they are fidgeting''': '''{"Dim": "Allo-Feedback", "Func": "Allo-Negative"} {"Dim": "Task", "Func": "Set Question & Inform"}''',
'''Assistant-1: Kids are great at fidgeting, so let's make sure we give them permission to fidget.''': '''{"Dim": "Task", "Func": "Inform & Instruct"}''',
'''Assistant-2: Yeah, definitely. What works for you personally? What works for other people?''': '''{"Dim": "Auto-Feedback", "Func": "Auto-Positive"} {"Dim": "Task", "Func": "Set Question"}''',
	'''Answer''': '''fill either 1 or 2''',
	'''Explanation''': '''fill your explanation here'''
}

Example Dialog 2 - 
Human: what are some good novels for a 9 year old?
Assistant: That depends on the 9 year old, but if they like thrillers or mysteries then maybe something like "The Body in the Bog" or "The Boy Who Followed Ripley." For more modern, female-written fiction I might recommend "Jane Eyre" or "Frankenstein."
Human: That sounds cool
Assistant-1: I'm a programmer, so I'm really into good programming texts, like "The Practice of Programming" or "JavaScript: The Good Parts." But that might be a bit too much for a 9 year old, so
Assistant-2: Do you want some quotes?

Example Output 2 - 
{
'''Human: what are some good novels for a 9 year old?''': '''{"Dim": "Task", "Func": "Set Question"}''',
'''Assistant: That depends on the 9 year old, but if they like thrillers or mysteries then maybe something like "The Body in the Bog" or "The Boy Who Followed Ripley." For more modern, female-written fiction I might recommend "Jane Eyre" or "Frankenstein."''': '''{"Dim": "Answer & Suggest"}''',
'''Human: That sounds cool''': '''{"Dim": "Allo-Feedback", "Func": "Allo-Positive"}''',
'''Assistant-1: I'm a programmer, so I'm really into good programming texts, like "The Practice of Programming" or "JavaScript: The Good Parts." But that might be a bit too much for a 9 year old, so''': '''{"Dim": "Task", "Func": "Inform"} {"Dim": "Own Communication Management", "Func": "Retraction"}''',
'''Assistant-2: Do you want some quotes?''': '''{"Dim": "Task", "Func": "Offer & Propositional Question"}''',
	'''Answer''': '''fill either 1 or 2''',
	'''Explanation": '''fill your explanation here'''
}


Dialog - 
"""