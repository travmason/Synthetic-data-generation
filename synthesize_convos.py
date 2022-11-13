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
base_prompt = open_file('syn_prompt2.txt')


def gpt3_completion(wdir, prompt, topic, engine='text-davinci-002', temp=1, top_p=1.0, tokens=3500, freq_pen=0.0, pres_pen=0.5, stop=['<<END>>']):
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
            with open(wdir + '/%s' % filename, 'w') as outfile:
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

def compare_cosine(doclist):
    count_vect = CountVectorizer()
    for a, b in itertools.combinations(doclist, 2):
        corpus = [a,b]
        X_train_counts = count_vect.fit_transform(corpus)
        pd.DataFrame(X_train_counts.toarray(),columns=count_vect.get_feature_names(),index=['Document 1','Document 2'])
        vectorizer = TfidfVectorizer()
        trsfm=vectorizer.fit_transform(corpus)
        pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['Document 1','Document 2'])
        cosine_similarity(trsfm[0:1], trsfm)

if __name__ == '__main__':
    # default prompt attributes
    brothers = "no brothers"
    sisters = "one sister"
    severity = "mild"
    alone = "lives"
    topic = "depression"

    topics = open_file('topics.txt').splitlines()
    vars_df = pd.read_csv('vars.csv')

    first_utterance = open_file('utterances.txt').splitlines()
    utterance_loop = len(first_utterance)

    directory = 'gpt3_logs'
    loops = 0
    utt_loop = 0
    raw_utterance = "\nUser: <<UTT>>\nDaniel:"
    prompt_arr = {
        'Prompt':[], 
        'Topic': [], 
        'Utterance': [],
        'Response': []
    }

    #create a directory for this run in gpt3_logs
    filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))
    print(filelist)
    myList = [i.split('.')[0] for i in filelist]
    print(myList)
    working_dir = str(int(max(myList))+1) + '.run'
    os.mkdir(directory + '\\' + working_dir)
    print('Creating %s\n' % working_dir)

    #set the new working directory based on the new working directory name
    directory = directory + '\\' + working_dir

    # for severity in vars_df["severity"]:
    #     if type(severity) != str:
    #         exit()
    #     print("Severity: %s\n" % severity)
    for topic in vars_df["topic"]:
        if type(topic) != str:
            break
        print("Topic: %s\n" % topic)

        for utterance in first_utterance:
            if type(utterance) != str:
                break
            print("Utterance: %s\n" % utterance)
            prompt = base_prompt.replace('<<TOPIC>>', topic)
            utterance = raw_utterance.replace('<<UTT>>', utterance)
            prompt_arr['Prompt'].append(prompt)            
            prompt_arr['Topic'].append(topic)            
            prompt_arr['Utterance'].append(utterance)
            prompt += utterance
            prompt = str(uuid.uuid4()) + '\n' + prompt

            response = gpt3_completion(directory, prompt, topic)
            prompt_arr['Response'].append(response)

        print('\n---------------------------------\n')
        df = pd.DataFrame(data=prompt_arr)
        print('df:')
        print(df)
        prompt_arr = {
            'Prompt':[], 
            'Topic': [], 
            'Utterance': [],
            'Response': []
        }
        df.to_json(directory + "\\%s__output.json" % (topic))

            