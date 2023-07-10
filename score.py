import os
import openai
from time import time,sleep
from dotenv import load_dotenv
import pandas as pd
import json
import uuid
import math
import itertools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
base_prompt = open_file('score_prompt.txt')

def gpt4_completion(wdir, prompt, topic, engine='gpt-4', temp=1, top_p=1.0, tokens=3500, freq_pen=0.0, pres_pen=0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    while True:
        try:
            messages=[{"role": "user", "content": f'{prompt}'}]

            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['message']['content'].strip()
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print(f'Error communicating with OpenAI: {oops}\n')
            sleep(0.25)

def quality_check(run):
    filelist = filter(lambda x: (x.endswith('.json')), os.listdir(f'gpt3_logs/{run}.run'))
    for file in filelist:
        current_topic = open(file, 'r')
        print(f'{current_topic}\n')

def score(filepath):
    prompt_arr = {
        'Prompt':[], 
        'Response': []
    }

    with open(filepath, 'r') as file:
        conversation = file.read()
    prompt = base_prompt.replace('<<CONVERSATION>>', conversation)
    prompt_arr['Prompt'].append(prompt)            
    # print(f'Prompt: {prompt}\n')
    response = gpt4_completion("", prompt, "score")
    prompt_arr['Response'].append(response)

    return response

if __name__ == '__main__':
    directory = 'gpt3_logs'

    # Create a list of all the files in the current directory
    # we're going to use this to create a new directory for the current run
    # filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))

    # # Find the highest numbered directory
    # highest_number = 0
    # for file in filelist:
    #     try:
    #         number = int(file.rstrip('.run'))
    #         if number > highest_number:
    #             highest_number = number
    #     except ValueError:
    #         pass  # Ignore if the file name is not a number

    # # Create a new directory with a number +1 higher than the highest
    # new_dir_number = highest_number + 1
    # new_directory = os.path.join(directory, f"{new_dir_number}.run")

    # print('Creating %s\n' % new_directory)

    # os.makedirs(new_directory)

    # #set the new working directory based on the new working directory name
    # directory = new_directory

    # print(f'')

    # Directory path containing the JSON files
    directory_path = 'gpt3_logs/41.run'

    response = []
    loopcount = 0

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}")
            # Process the file
            response.append(score(file_path).strip())
            print(f"Score: {response[-1]}")
            print(f"Loopcount: {loopcount}")
        # if loopcount > 4:
        #     break
        # loopcount += 1
    with open('gpt3_logs/%s' % 'score.log', 'a', encoding='utf-8') as f:
        json.dump(response, f, indent=4)


        