# flask, tensorflow_hub, tensorflow-cpu, sklearn, gunicorn

from flask import Flask, render_template, request
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


# loading the pre-trained USE model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("model is now loaded")

def embed(input):
  return model(input)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/similarity", methods=['POST'])
def similarity():
    sent1 = np.array(embed([request.form.get('sent1').strip()]))
    sent2 = np.array(embed([request.form.get('sent2').strip()]))
    
    simi = cos_sim(sent1, sent2)[0][0]
    print(simi)

    return f"{str(round(simi, 2))}"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
