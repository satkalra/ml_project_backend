
from flask import Flask, jsonify
import pandas as pd
import random

app = Flask(__name__)

# Define a route for your API
df1 = pd.read_csv('./cluster1_6271.csv')
df2 = pd.read_csv('./cluster2_1311.csv')


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


if __name__ == '__main__':
    app.run(debug=False)