# Conversation generator for counsellor / patient analysis

I have synthasis and scoring going on here.

## Synthesis

Grunt work is done through the `synthesize_convos.py` script. Looping through topics.txt to generate different conversation topics and generating two way transcripts beterrn Daniel (AI) and a Counsellor (Human).

TODO:
- How do we generate longer conversations?
- Output formatting
- Embed more personal information into Daniel (is this how we do that? Generate Finetune data and have it in the model itself?)
- more..

## Score

Look at `score.py` and `score_view.py`. I'm using an assessment criteria input through `score_prompt_counselling_competancys.txt` (based on a set of competancies from clinical assessment)
Pulls in conversations generated from the synthasise convos script and outputs to a set of graphs.

TODO:
- working on output format consistancy (new data analysis in GPT-4 Turbo should help here)
- Data Analysis from 'native' gpt-4 release November to improve results?
- Change criteria based on need, remains of code should stay valid.

## Example

```
User: Hi Daniel, how can I best help you today?
Daniel: I don't know. I just feel really down all the time.
User: That's tough. Can you tell me more about why you're feeling down?
Daniel: I don't know. Everything just seems pointless.
User: That sounds very discouraging. Can you think of anything that brings you joy?
Daniel: Not really.
User: That's understandable. It sounds like you're going through a lot.
Daniel: Yeah.
User: I'm here for you, though. We'll get through this together.
Daniel: Thanks.
User: Do you want to talk about what's going on in your life that might be causing this depression?
Daniel: I don't know. My brothers are both doing so well and I feel like I can't measure up.
User: That sounds like a lot of pressure you're putting on yourself.
Daniel: Yeah.
User: It's okay to not be perfect. We all have different strengths and weaknesses.
Daniel: I guess.
User: Is there anything else on your mind?
Daniel: No. I just feel really down all the time.
User: That's tough. Can you tell me more about why you're feeling down?
Daniel: I don't know. Everything just seems pointless.
User: That sounds very discouraging. Can you think of anything that brings you joy?
Daniel: Not really.
User: That's understandable. It sounds like you're going through a lot.
Daniel: Yeah.
User: I'm here for you, though. We'll get through this together.
Daniel: Thanks.
```


