from flask import Flask, request, make_response
from flask_cors import CORS
from model import llm, hfe

app = Flask(__name__)
cors = CORS(app, resources={r"/llama/*": {"origins": "*"}})

@app.route("/llama/query", methods=["POST"])
def query():
    context = request.json.get('context', "")
    question = request.json.get('question', "")
    answer = llm(prompt="Write an answer within 150 words "
    "for the question below based on the provided context. "
    "If the context provides insufficient information, "
    'reply "I cannot answer". '
    "Answer in an unbiased, comprehensive, and formal tone. "
    "If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences.\n\n"
    f"Context: {context}\n"
    f"Question: {question}\n"
    "Answer: ")
    
    return make_response({"answer": answer}, 200)
    
if __name__ == "__main__":
    app.run()