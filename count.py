import os
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

text = ''
stopwords = set(STOPWORDS)
stopwords.update(['User', 'Daniel', 'David', 'really', 'know'])

dir = 'gpt3_logs'
for filename in os.listdir(dir):
    f = os.path.join(dir, filename)
    if os.path.isfile(f):
        data = open_file(f)
        sto = data.split('RESPONSE:')
        #print('Len sto: %s' % len(sto))
        if len(sto) > 1:
            #print('Response: %s' % sto[1])
            #print(sto[1].splitlines())
            #text += sto[1]
            print('Length (lines): %s' % len(sto[1].splitlines()))

# wordcloud = WordCloud(width=1000, height=500, stopwords=stopwords, max_words=500, background_color="white").generate(text)
# plt.figure( figsize=(20,10), facecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()

# print('lines = %s' % str(len(text.splitlines(True))))
# returned_lines = str(len(text.splitlines(True)))
# filename = '%s_gpt3.txt' % time()
# with open('gpt3_logs/%s' % filename, 'w') as outfile:
#     outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
# response_info = '{"topic" : "%s",\nengine" : "%s",\ntemp" : "%s",\ntop_p" : "%s",\nfreq_pen" : "%s",\npres_pen" : "%s",\nreturned lines" : "%s" }' % (topic, engine, temp, top_p, freq_pen, pres_pen, returned_lines)
# data = response_info.split('\n')
# with open('gpt3_logs/%s' % 'data.log', 'a', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
