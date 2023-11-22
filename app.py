from flask import Flask, request, make_response
from flask_cors import CORS
from model import llm

app = Flask(__name__)
cors = CORS(app, resources={r"/llama/*": {"origins": "*"}})

@app.route("/llama", methods=["POST"])
def query():
    context = request.json.get('context', "")
    question = request.json.get('question', "")
    lang = request.json.get('lang', "vi")
    
    if lang == "en":
        answer = llm(
        "Write an answer within 100 words for the question below based on the provided context. "
        "If the context provides insufficient information, reply \"I cannot answer\". "
        "Answer in an unbiased, comprehensive, and formal tone. "
        "If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer: ") # "<<SYS>> You are an intelligent agent who can answer questions based on context. <<SYS>>\n"
        # answer = output[0]["generated_text"]
    elif lang == "vi":
        answer = llm(
        "Hãy trả lời câu hỏi bên dưới với các thông tin được cung cấp trong phần ngữ cảnh. "
        "Nếu trong ngữ cảnh không có đủ thông tin, hãy trả lời \"Tôi không biết\". "
        "Câu trả lời phải đầy đủ thông tin, có giải thích và không nhiều hơn 100 từ.\n\n"
        f"Ngữ cảnh: {context}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời: ") # "<<SYS>> Bạn là agent thông minh có thể trả lời câu hỏi theo ngữ cảnh. <<SYS>>\n"
        # answer = output[0]["generated_text"]
    else:
        answer = "I do not understand your question!"
        
    answer = answer.split('\n\n')[0]
        
    return make_response({"answer": answer}, 200)
    
@app.route("/llama/api", methods=["POST"])
def api():
    prompt = request.json.get('prompt', "")
    lang = request.json.get('lang', "en")
    
    if lang == "en":
        answer = llm(prompt)
    elif lang == "vi":
        answer = llm(prompt)
    
    return make_response({"answer": answer}, 200)

if __name__ == "__main__":
    app.run()