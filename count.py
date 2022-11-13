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

def topic_model(text):
    #measure how well a conversation sticks to topic
    # Load the regular expression library

    # Remove punctuation
    text["Response_proc"] = \
    text["Response"].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    text["Response_proc"] = \
    text["Response_proc"].map(lambda x: x.lower())
    # Print out the first rows of papers
    print('Prompt : \n%s\n' % text["Prompt"].head())
    print('Topic : \n%s\n' % text["Topic"].head())
    print('Utterance : \n%s\n' % text["Utterance"].head())
    print('Response_proc : \n%s\n' % text["Response_proc"].head())

    # # Import the wordcloud library
    # from wordcloud import WordCloud
    # # Join the different processed titles together.
    # long_string = ','.join(list(text['Response_proc'].values))
    # stop_words = ["user", "david"] + list(STOPWORDS)
    # # Create a WordCloud object
    # wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # # Generate a word cloud
    # wordcloud.generate(long_string)
    # # Visualize the word cloud
    # wordcloud.to_image().show()

    import gensim
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['user', 'david', 'hi'])
    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]
    data = text.Response_proc.values.tolist()
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

    import pyLDAvis.gensim_models
    import pickle 
    import pyLDAvis
    # Visualize the topics
    LDAvis_data_filepath = os.path.join('./gpt3_logs/ldavis_prepared_'+str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './gpt3_logs/ldavis_prepared_'+ str(num_topics) +'.html')
    LDAvis_prepared


if __name__ == '__main__':
    df = pd.DataFrame()
    directory = 'gpt3_logs'

    #create a directory for this run in gpt3_logs
    filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))
    print(filelist)
    myList = [i.split('.')[0] for i in filelist]
    print(myList)
    working_dir = str(max(myList)) + '.run'
    print('Loading from %s\n' % working_dir)

    #set the new working directory based on the new working directory name
    directory = directory + '\\' + working_dir

    filename = 'output.json'
    #f = os.path.join(dir, filename)
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            print('Loading %s\n' % filename)
            f = os.path.join(directory, filename)
            if df.empty:
                df = pd.read_json(f)
            else:
                df = pd.concat([df, pd.read_json(f)], axis=0)
    print(df.info)
    print(df.dtypes)
    print(df.describe())
    for d in df["Response"]:
        print('Length (lines): %s' % len(d.splitlines()))
        #print("\n\n%s\n" % d)
    topic_model(df)

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
