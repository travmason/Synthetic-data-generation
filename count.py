import os
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re

import matplotlib.pyplot as plt

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

text = ''
stopwords = set(STOPWORDS)
stopwords.update(['User', 'Daniel', 'David', 'really', 'know'])

dir = 'gpt3_logs'
# for filename in os.listdir(dir):
#     f = os.path.join(dir, filename)
#     if os.path.isfile(f):
#         data = open_file(f)
#         sto = data.split('RESPONSE:')
#         #print('Len sto: %s' % len(sto))
#         if len(sto) > 1:
#             #print('Response: %s' % sto[1])
#             #print(sto[1].splitlines())
#             #text += sto[1]
#             print('Length (lines): %s' % len(sto[1].splitlines()))

def topic_model(text):
    #measure how well a conversation sticks to topic
    # Load the regular expression library

    # Remove punctuation
    text['paper_text_processed'] = \
    text['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    text['paper_text_processed'] = \
    text['paper_text_processed'].map(lambda x: x.lower())
    # Print out the first rows of papers
    text['paper_text_processed'].head()

    # Import the wordcloud library
    from wordcloud import WordCloud
    # Join the different processed titles together.
    long_string = ','.join(list(papers['paper_text_processed'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()

    import gensim
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]
    data = text.paper_text_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    print(data_words[:1][0][:30])

    import gensim.corpora as corpora
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(corpus[:1][0][:30])

    from pprint import pprint
    # number of topics
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    import pyLDAvis.gensim
    import pickle 
    import pyLDAvis
    # Visualize the topics
    pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
    LDAvis_prepared

filename = 'output.json'
f = os.path.join(dir, filename)
df = pd.read_json(f)
print(df.info)
print(df.dtypes)
print(df.describe())
for d in df["Response"]:
    print('Length (lines): %s' % len(d.splitlines()))
    #print("\n\n%s\n" % d)
topic_model(d)

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
