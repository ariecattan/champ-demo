# app.py
from flask import Flask, render_template, request, jsonify
import os
import pandas as pd 
import io 
import json
import collections 
from itertools import permutations, chain, combinations
import networkx as nx 

app = Flask(__name__)

def _get_clusters(data):
    clusters = collections.defaultdict(list)
    for kp in data["mentions"]:
        clusters[kp["clustId"][0]].append(int(kp["m_id"])) 
    all_clusters = list(clusters.values())
    return all_clusters


def _get_binary_relations(tree):
    binary_relations = []
    if 'children' not in tree:
        raise ValueError(tree)
    stack = tree['children'].copy()

    while stack:
        node = stack.pop()

        if 'children' in node:
            node_relations = []
            for child in node['children']:
                node_relations.append((node['id'], child['id']))
                stack.append(child.copy())
            binary_relations.extend(node_relations)

    return binary_relations

def _get_mention2cluster(topic):
    mention2cluster = {}
    for cluster_id, cluster in enumerate(topic["clusters"]):
      for kp in cluster:
        mention2cluster[kp] = cluster_id

    return mention2cluster

def _get_topic_graph(topic):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(topic["clusters"]))))
    G.add_edges_from(topic["relations"])
    return G 


def evaluate_topic(gold_topic, sys_topic):
    gold_tree = _get_topic_graph(gold_topic)
    system_tree = _get_topic_graph(sys_topic)

    m2c_gold = _get_mention2cluster(gold_topic)
    m2c_system = _get_mention2cluster(sys_topic)

    tp, tn, fp, fn = 0, 0, 0, 0

    for a, b in permutations(list(range(len(gold_topic["mentions"]))), 2):
      # same cluster in both gold and system 
      is_gold_link = m2c_gold[a] == m2c_gold[b] or nx.has_path(gold_tree, m2c_gold[a], m2c_gold[b])
      is_sys_link = m2c_system[a] == m2c_system[b] or nx.has_path(system_tree, m2c_system[a], m2c_system[b])

      if is_gold_link and is_sys_link:
        tp += 1 
      elif is_gold_link and not is_sys_link:
        fn += 1
      elif not is_gold_link and is_sys_link:
        fp += 1
      else:
        tn += 1


    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    return {
      "precision": precision,
      "recall": recall,
      "f1": 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    }


# Your compute_iaa function, replace this with your actual implementation
def compute_iaa(files):
    # Your implementation here
    # Process the files and compute the inter-annotator agreement
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 27],
        'City': ['New York', 'London', 'Paris']
    }
    df = pd.DataFrame(data)
    return df



def process_inputs(files):
    dataframes = []
    for i, file in enumerate(files):
        json_data = json.loads(file.read())
        tokens = [x["text"] for x in json_data["tokens"]]
        mentions = [" ".join(tokens[mention["start"]:mention["end"]+1]) for mention in json_data["mentions"]]

        data = {
            "id": json_data["id"],
            "name": json_data["name"] if "name" in json_data else str(i),
            "mentions": mentions,
            "clusters": _get_clusters(json_data),
            "relations": _get_binary_relations(json_data["tree"][0])
        }

        dataframes.append(data)

    return pd.DataFrame(dataframes)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    # Get the uploaded files from the request
    uploaded_files = request.files.getlist('files')

    df = process_inputs(uploaded_files)

    all_possible_pairs = [f'{a}-{b}' for a, b in combinations(df["name"].unique(), r=2) if a != b]
    topics = df["id"].unique()

    # init iaa matrix
    scores = collections.defaultdict(dict)
    for topic in topics:
       for annotator_pair in all_possible_pairs:
          scores[topic][annotator_pair] = None

    # compute iaa for each topic separately
    for topic_id, topic_annotation in df.groupby("id"):
       topic_pairs = [f'{a}-{b}' if f'{a}-{b}' in all_possible_pairs else f'{b}-{a}' for a, b in combinations(topic_annotation["name"], r=2)]
       topic_annotation = topic_annotation.set_index("name")
       for pair in topic_pairs:
          a, b = pair.split("-")
          pair_id = f'{a}-{b}' if f'{a}-{b}' in topic_pairs else f'{b}-{a}'
          iaa = evaluate_topic(topic_annotation.loc[a], topic_annotation.loc[b])
          scores[topic_id][pair_id] = iaa["f1"]

    df_scores = pd.DataFrame(scores).T
    
    df_html = df_scores.to_html(classes='table table-bordered table-hover')
    
    return jsonify({"table": df_html})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)