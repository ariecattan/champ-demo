# app.py
from flask import Flask, render_template, request, jsonify
import os
import pandas as pd 

app = Flask(__name__)

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("files")
    # Process the uploaded files and call the compute_iaa function
    df = compute_iaa(files)
    # Convert the DataFrame to HTML table representation
    df_html = df.to_html(classes='table table-bordered table-hover', index=False)
    return jsonify({"table": df_html})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)
