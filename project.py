import os
from flask import Flask, render_template, request,jsonify
from groq import Groq  
import pandas as pd
import joblib

app = Flask(__name__)

# 1. 配置 Groq 客户端 (在 Codespaces Settings 中设置环境变量)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# 2. 加载信用评估模型
# 注意：请确保将模型文件命名为 credit_model.pkl 并放在 models 文件夹下
try:
    model = joblib.load('models/credit_model.pkl')
except:
    model = None
    print("Note: If the model file is not found, simulation model will be used.")


@app.route("/", methods=["GET","POST"])
def index():
    return render_template('index.html')

# 模块一：AI 聊天机器人 (基于 Groq )
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a regional bank assistant, responsible for assisting customers in handling their transactions."},
                {"role": "user", "content": user_input}
            ],
        )
        bot_response = completion.choices[0].message.content
        return jsonify({"status": "success", "response": bot_response})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# 模块二：贷款审批（机器学习）
@app.route('/predict_credit', methods=['POST'])
def predict_credit():
    data = request.json
    
    # 构建输入数据帧 (需与您训练时的特征顺序一致)
    input_data = {
        'duration': [int(data.get('duration', 0))],
        'amount': [int(data.get('amount', 0))],
        'age': [int(data.get('age', 0))]
        # ... 根据您的模型添加更多特征
    }
    df = pd.DataFrame(input_data)
    
    if model:
        prediction = model.predict(df)[0]
        # 假设 1 为 Good, 0 为 Bad
        result = "Approved (with good credit)" if prediction == 1 else "Rejected (high risk)"
    else:
        # 兜底模拟逻辑
        result = "Approved" if int(data.get('amount', 0)) < 10000 else "Requireing further manual review"
    
    return jsonify({"result": result})

if __name__ == '__main__':
    # host='0.0.0.0' 是在 Codespaces 中正常预览的关键
    app.run(host='0.0.0.0', port=5000, debug=True)