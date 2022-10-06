import os
import openai
from time import time,sleep
from dotenv import load_dotenv

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

def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=3500, freq_pen=0.0, pres_pen=0.5, stop=['<<END>>']):
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
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(0.25)


if __name__ == '__main__':
    topics = open_file('topics.txt').splitlines()
    loops = 0
    for topic in topics:
        if loops < 3:
            print(topic)
            prompt = open_file('syn_prompt2.txt').replace('<<TOPIC>>', topic)
            response = gpt3_completion(prompt)
            outtext = 'Daniel: %s' % response
            print(outtext)
            tpc = topic.replace(' ', '')[0:15]
            save_convo(outtext, tpc)
            loops += 1
        else:
            exit()
        