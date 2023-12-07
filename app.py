
from flask import Flask, jsonify, request
import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import re

load_dotenv()


OPENAI_KEY=os.getenv('OPENAI_KEY')
debug=os.getenv('DEBUG')

client = OpenAI(api_key=OPENAI_KEY)

def clean_text(text):
    try:
        text = re.sub('\xa0', ' ', text)
        text = re.sub('\u200b', ' ', text)
        text = re.sub('\u200b', ' ', text)
        text = re.sub('\n', ' ', text)
        text = re.sub('\t', ' ', text)
        text = re.sub('\r', ' ', text)
        return text.strip()
    except:
        return np.nan

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model='text-embedding-ada-002'):
    text = text.replace("\n", " ")

    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_similarity(question, data, n=4):
    ques_embed = get_embedding(question)
    data['cosine_similarity'] = data['embedding'].apply(lambda x:cosine_similarity(x, ques_embed))
    context = ' '.join(data.sort_values('cosine_similarity', ascending=False).head(n)[['text']].values[0])

    response = client.completions.create(model = "gpt-3.5-turbo-instruct", 
                                    temperature=0, 
                                       prompt = f"""Your name is counsel. Conusel is designed to answer question about research in machine learning domain. Cousel talks politely and greets everyone. Keep answers within the context and give refrences for your answer  
                                                    You can answer basic of Machine learning question, consider them in context.

                                                    
                                                   Question: {question}
                                                   
                                                   Context: {context}


                                                   If the question is out of context, politely refuse to answer.
                                                                                                      
                                                    Do Not make answers on your own. Do not hallucinate. 

                                                   """,
                                        max_tokens=1024
                                       )
                                        
    return response


app = Flask(__name__)

# Define a route for your API
df1 = pd.read_csv('./cluster1_6271.csv')
df2 = pd.read_csv('./cluster2_1311.csv')

answer_df1 = pd.read_json('./embedding1.json')
answer_df2 = pd.read_json('./embedding2.json')


def get_sample_by_cluster(df,cluster_number,samples):
    cluster_df = df[df['cluster_label'] == cluster_number]
    return cluster_df.sample(n=samples)


@app.route('/api/get-sample-cluster-1', methods=['GET'])
def get_data_cluster_1():
    random_cluster = random.randint(-1, 10)
    sample = get_sample_by_cluster(df1,random_cluster,2)
    return jsonify({'text': sample['documents'].to_list(), 'keywords': sample['keywords_list'].to_list()})

@app.route('/api/get-sample-cluster-2', methods=['GET'])
def get_data_cluster_2():
    random_cluster = random.randint(-1, 4)
    sample = get_sample_by_cluster(df2,random_cluster,2)
    return jsonify({'text': sample['Cleaned_Data'].to_list(), 'keywords': sample['keywords_list'].to_list()})

@app.route('/api/answer', methods=['POST'])
def get_chatbot_answer():
    data = request.get_json()
    question = data.get('question')
    choice = data.get('embedding')
    df = pd.DataFrame()

    if choice == 1:
        df = answer_df1
    else:
        df = answer_df2

    answer = get_similarity(question,df).choices[0].text

    return jsonify({'answer': clean_text(answer)})


if __name__ == '__main__':
    app.run(debug=debug)