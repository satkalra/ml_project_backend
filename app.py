
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
                                       prompt = f"""Your name is Counsel. Counsel is a chatbot that can answer questions about machine learning.
                                                    Counsel knows names of authors, title, publication name and year of papers related to papers.
                                                    Counsel knows the concept of time and date.
                                                    
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
answer_df1 = pd.read_json('./embedding1.json')
answer_df2 = pd.read_json('./embedding2.json')


def get_sample_by_cluster(df,cluster_number,samples):
    cluster_df = df[df['cluster_label'] == cluster_number]
    return cluster_df.sample(n=samples)

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