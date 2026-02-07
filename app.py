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
# 注意：请确保将模型文件命名为 loan_model.pkl 并放在 models 文件夹下
try:
    model = joblib.load('models/loan_model.pkl')
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
# 修复变量名并添加简单校验
@app.route('/predict_credit', methods=['POST'])
def predict_credit():
    data = request.json
    try:
        # 修复：与前端字段名保持一致
        loan_amount = int(data.get('amount', 0))
        loan_rate = float(data.get('rate', 0))
        person_income = int(data.get('person_income', 0))
    except (ValueError, TypeError):
        return jsonify({"result": "Invalid input data"}), 400

    # 添加参数验证
    if loan_amount <= 0 or loan_rate < 0 or person_income <= 0:
        return jsonify({"result": "Invalid input: Please enter valid positive numbers"}), 400

    if model:
        # 2. 构建包含三个特征的数据帧 (注意顺序必须与训练时一致)
        input_data = {
            'loan_amnt': [loan_amount],
            'loan_int_rate': [loan_rate],
            'person_income': [person_income]
        }
        df = pd.DataFrame(input_data)
        
        prediction = model.predict(df)[0]
        result = "Approved(acceptable credit)" if prediction == 0 else "Rejected(high risk)"
        print(f"[ML Model] Prediction: {prediction}, Input: Loan=${loan_amount}, Rate={loan_rate}%, Income=${person_income}")
    else:
        # 模型未加载时的兜底逻辑
        result = " Manual review required"
        print(f"[WARNING] Model not loaded. Returning: {result}")
    
    return jsonify({"result": result})

if __name__ == '__main__':
    # host='0.0.0.0' 是在 Codespaces 中正常预览的关键
    app.run(host='0.0.0.0', port=5000, debug=True)