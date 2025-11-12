import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_core._api.deprecation')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from flask import Flask, render_template, request, jsonify, session
from src.rag_pipeline import get_rag_response
from langchain_core.messages import HumanMessage, AIMessage
import time

app = Flask(__name__, static_folder='Static', template_folder='Templates')
app.secret_key = os.urandom(24)

try:
    from src.rag_pipeline import get_initialized_components
    get_initialized_components()
except Exception as e:
    print(f"Pre-warming failed {e}")

@app.route('/')
def index():
    session.clear()
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    
    user_query = request.json.get('message', '').strip()
    
    if not user_query:
        return jsonify({"answer": "Please enter a question.", "sources": []})
    
    if 'chat_history' not in session:
        session['chat_history'] = []

    langchain_history = []
    for msg in session['chat_history']:
        if msg['role'] == 'user':
            langchain_history.append(HumanMessage(content=msg['content']))
        else:
            langchain_history.append(AIMessage(content=msg['content']))

    try:
        rag_result = get_rag_response(user_query, langchain_history)
        
        session['chat_history'].append({"role": "user", "content": user_query})
        session['chat_history'].append({"role": "assistant", "content": rag_result['answer']})
        session.modified = True
        
        response_time = time.time() - start_time
        print(f"Query processed in {response_time:.2f}s")
        
        return jsonify(rag_result)
        
    except Exception as e:
        print(f"Chat route error: {e}")
        return jsonify({
            "answer": "Sorry, I encountered an error processing your request. Please try again.",
            "sources": []
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)