import os
import openai
from time import time,sleep
from dotenv import load_dotenv
import pandas as pd
import json
import uuid

try:
    load_dotenv()  # take environment variables from .env.
except Exception as oops:
    print("Issue with load_dotenv:" + oops)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_convo(text, topic):
    with open('finetuning/%s_%s.txt' % (topic, time()), 'w', encoding='utf-8') as outfile:
        outfile.write(text)


openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_version = os.getenv("PROMPT_VERSION")
base_prompt = open_file('syn_prompt2.txt')


def gpt3_completion(prompt, topic, engine='text-davinci-002', temp=1, top_p=1.0, tokens=3500, freq_pen=0.0, pres_pen=0.5, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            print('lines = %s' % str(len(text.splitlines(True))))
            returned_lines = str(len(text.splitlines(True)))
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            response_info = '{"topic" : "%s",\nengine" : "%s",\ntemp" : "%s",\ntop_p" : "%s",\nfreq_pen" : "%s",\npres_pen" : "%s",\nreturned lines" : "%s" }' % (topic, engine, temp, top_p, freq_pen, pres_pen, returned_lines)
            data = response_info.split('\n')
            with open('gpt3_logs/%s' % 'data.log', 'a', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(0.25)


if __name__ == '__main__':
    topics = open_file('topics.txt').splitlines()

    first_utterance = open_file('utterances.txt').splitlines()
    utterance_loop = len(first_utterance)

    loops = 0
    utt_loop = 0
    raw_utterance = "\nUser: <<UTT>>\nDaniel:"
    prompt_arr = {
        'Prompt':[], 
        'Topic': [], 
        'Utterance': [],
        'Response': []
    }

    for topic in topics:
        for utterance in first_utterance:
            if loops < 3:
                prompt = base_prompt.replace('<<TOPIC>>', topic)
                utterance = raw_utterance.replace('<<UTT>>', utterance)
                prompt_arr['Prompt'].append(prompt)            
                prompt_arr['Topic'].append(topic)            
                prompt_arr['Utterance'].append(utterance)
                prompt += utterance
                prompt = str(uuid.uuid4()) + '\n' + prompt

                response = gpt3_completion(prompt, topic)
                prompt_arr['Response'].append(response)

                # outtext = 'Daniel: %s' % response
                # print(outtext)
                # tpc = topic.replace(' ', '')[0:15]
                # save_convo(outtext, tpc)
                loops += 1
            else:
                #print(prompt_arr)
                print('\n---------------------------------\n')
                df = pd.DataFrame(data=prompt_arr)
                print('df:')
                print(df)
                df.to_json(r'gpt3_logs\output.json')
                exit()
        