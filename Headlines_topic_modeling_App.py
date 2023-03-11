# Importing important libraries

import numpy as np
import pandas as pd

import re, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

import pickle
import streamlit as st
import warnings

warnings.filterwarnings('ignore')


#Defining Helper Functions
def lower_cased_punctuation_removal(text):
    for punc in string.punctuation:  #String.punctuation has all punctation marks
        text = text.replace(punc, '') #Reomval of that punctuation from each text
        s = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    return text.lower()  #Converting the text to lowercase

def remove_stop_words(text):
    stop_words = stopwords.words('english')  #Creates a list of all stop words available in english from stopwords nltk library
    text_words = text.split()  #Creates list of individual words in text

    result_words  = [word for word in text_words if word not in stop_words]   #Keeps all non-stop words
    resultant_text = ' '.join(result_words)  #joins remaining words back into a sentence

    return resultant_text



def stemming_text(text):
    stemmer = PorterStemmer()  #Quite popular stemmer and is best for most use-cases.
    text_words = text.split()   #Splits the text into words
    stemmed_words  = [stemmer.stem(word) for word in text_words]   #Stems each word from the list
    stemmed_text = ' '.join(stemmed_words)  #Convert the words back into a sentence

    return stemmed_text


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()  #Quite popular Lemmatizer and is best for most use-cases.
    text_words = text.split()   #Splits the text into words
    lemmatized_words  = [lemmatizer.lemmatize(word) for word in text_words]   #Lemmatizes each word from the list
    lemmatized_text = ' '.join(lemmatized_words)  #Convert the words back into a sentence

    return lemmatized_text









# Defining Query Preprocessing functions

def query_preprocess_bow(text):
    processed_text = stemming_text(remove_stop_words(lower_cased_punctuation_removal(text)))
    return processed_text


def query_preprocess_W2V_Glove(text):
    processed_text = word_tokenize(lemmatize_text(lower_cased_punctuation_removal(text)))
    return processed_text


def query_preprocess_custom_W2V(text):
    processed_text = word_tokenize(lower_cased_punctuation_removal(text))
    return processed_text


def query_preprocess_lsa(text):
    processed_text = lower_cased_punctuation_removal(text)
    return processed_text


# Creating headline retrieval function:

def find_top_n_headlines_bow(bow_processed_df, bow_emb_array, query, count_vec, n=5):
    processed_query = [query_preprocess_bow(query)]
    query_emb_array = count_vec.transform(processed_query).toarray()

    cosine_sim_matrix = cosine_similarity(query_emb_array, bow_emb_array)

    x = np.flip(np.argsort(cosine_sim_matrix, axis=1)[0, -n:])

    similar_headlines_df = bow_processed_df.iloc[x][['headline']]
    similar_headlines_df['original_headlines'] = similar_headlines_df['headline']
    similar_headlines_df['similarity'] = cosine_sim_matrix[0, x]

    return similar_headlines_df[['headline', 'similarity']].reset_index(drop=True)


def find_top_n_headlines_W2V_Glove(W2V_df, query, W2V_or_glove_model, W2V_clustering_model, n=5):
    processed_query = query_preprocess_W2V_Glove(query)
    words = set(W2V_or_glove_model.index_to_key)

    text_embeddings = np.array([W2V_or_glove_model[i] for i in processed_query if i in words])

    sentence_embedding = np.mean(text_embeddings, axis=0).reshape(1, -1)

    sentence_embedding = np.array(sentence_embedding, dtype=np.double)

    cluster_label = W2V_clustering_model.predict(sentence_embedding)

    W2V_df_cluster = W2V_df[W2V_df.cluster_label == cluster_label[0]]

    W2V_emb_array = np.stack(W2V_df_cluster.sentence_emb.values)

    cosine_sim_matrix = cosine_similarity(sentence_embedding, W2V_emb_array)

    x = np.flip(np.argsort(cosine_sim_matrix, axis=1)[0, -n:])

    similar_headlines_df = W2V_df_cluster.iloc[x]
    similar_headlines_df['similarity'] = cosine_sim_matrix[0, x]

    return similar_headlines_df[['original_headlines', 'similarity']].reset_index(drop=True)


def find_top_n_headlines_LSA(lsa_df, query, lsa_tfidf_vect_model, lsa_svd_model, lsa_clustering_model, n=5):
    processed_query = query_preprocess_lsa(query)

    tf_idf_vector = lsa_tfidf_vect_model.transform([processed_query])

    sentence_embedding = lsa_svd_model.transform(tf_idf_vector)

    cluster_label = lsa_clustering_model.predict(sentence_embedding)

    lsa_cluster = lsa_df[lsa_df.cluster_label == cluster_label[0]]

    lsa_emb_array = np.stack(lsa_cluster.sentence_emb.values)

    cosine_sim_matrix = cosine_similarity(sentence_embedding, lsa_emb_array)

    x = np.flip(np.argsort(cosine_sim_matrix, axis=1)[0, -n:])

    similar_headlines_df = lsa_cluster.iloc[x]
    similar_headlines_df['similarity'] = cosine_sim_matrix[0, x]

    return similar_headlines_df[['original_headlines', 'similarity']].reset_index(drop=True)


def headline_finder(input_query, bow, word2vec, glove, custom_word2vec, lsa, n):
    models = []
    total_models = []
    all_similar_headlines = pd.DataFrame()

    if bow == True:
        bow_model = pickle.load(open('bag_of_words_count_vectorizer.pickle', 'rb'))
        bow_processed_headlines = pickle.load(open('deployment/embeddings_df/bow_processed_headlines.pickle', 'rb'))
        bow_emb_array = bow_model.transform(bow_processed_headlines.edited_headline).toarray()
        similar_headlines_bow = find_top_n_headlines_bow(bow_processed_headlines, bow_emb_array, input_query, bow_model,
                                                         n)
        models.append('Bag of Words')
        total_models.append(similar_headlines_bow)

    if word2vec == True:
        W2V_model = KeyedVectors.load('Word2Vec_model.bin')
        W2V_df = pickle.load(open('pretrained_Word2Vec_embeddings_df.pickle', 'rb'))
        W2V_clustering_model = pickle.load(open('pretrained_Word2Vec_clustering_model.pickle', 'rb'))
        similar_headlines_W2V = find_top_n_headlines_W2V_Glove(W2V_df, input_query, W2V_model, W2V_clustering_model, n)
        models.append('Pretrained-Word2Vec')
        total_models.append(similar_headlines_W2V)

    if glove == True:
        glove_model = KeyedVectors.load('glove_model.bin')
        glove_df = pickle.load(open('pretrained_GloVe_embeddings_df.pickle', 'rb'))
        glove_clustering_model = pickle.load(open('pretrained_GloVe_clustering_model.pickle', 'rb'))
        similar_headlines_glove = find_top_n_headlines_W2V_Glove(glove_df, input_query, glove_model,
                                                                 glove_clustering_model, n)
        models.append('Pretrained-GloVe')
        total_models.append(similar_headlines_glove)

    if custom_word2vec == True:
        custom_word2vec_model = KeyedVectors.load('Word2VecCustomTrained_model.bin')
        custom_W2V_df = pickle.load(open('custom_Word2Vec_embeddings_df.pickle', 'rb'))
        custom_W2V_clustering_model = pickle.load(open('custom_Word2Vec_clustering_model.pickle', 'rb'))
        similar_headlines_custom_W2V = find_top_n_headlines_W2V_Glove(custom_W2V_df, input_query, custom_word2vec_model,
                                                                      custom_W2V_clustering_model, n)
        models.append('Custom-Word2Vec')
        total_models.append(similar_headlines_custom_W2V)

    if lsa == True:
        lsa_tfidf_vect_model = pickle.load(open('tfidf_vect_LSA_model.pickle', 'rb'))
        lsa_svd_model = pickle.load(open('svd_LSA_model.pickle', 'rb'))
        lsa_df = pickle.load(open('LSA_embeddings_df.pickle', 'rb'))
        lsa_clustering_model = pickle.load(open('LSA_clustering_model.pickle', 'rb'))
        similar_headlines_lsa = find_top_n_headlines_LSA(lsa_df, input_query, lsa_tfidf_vect_model, lsa_svd_model,
                                                         lsa_clustering_model, n)
        models.append('LSA')
        total_models.append(similar_headlines_lsa)

    header = pd.MultiIndex.from_product([models,
                                         ['original_headlines', 'similarity']],
                                        names=['Model', 'Results'])

    all_results = pd.DataFrame(columns=header)

    for i in range(0, len(total_models)):
        all_results[models[i]] = total_models[i]

    return all_results




st.cache(suppress_st_warning= True, ttl=24*3600)
def main():

    # Giving title
    st.title('Relevant News Headline Finder')

    #Introduction to Web app
    st.write('This is a web app created to fetch relevant news headlines\
             Please provide a query and our different models will fetch its relevant result')

    text = st.text_input("Please input your search query")

    words = st.number_input('Number of top results', step=1, min_value=1, max_value=10)

    bow, word2vec, glove, custom_word2vec, lsa = st.columns(5)
    with bow:
        bow = st.checkbox('Bag of Words')
    with word2vec:
        word2vec = st.checkbox('Word2Vec (Pre-Trained)')
    with glove:
        glove = st.checkbox('GLoVe (Pre-Trained)')
    with custom_word2vec:
        custom_word2vec = st.checkbox('Word2Vec (Customized)')
    with lsa:
        lsa = st.checkbox('LSA')

    fetch_results = headline_finder(text, bow, word2vec, glove, custom_word2vec, lsa, words)

    if fetch_results.shape == (0, 0):
        st.write('Please select an option above.')
    else:
        st.write(fetch_results)


if __name__ == '__main__':
    main()
