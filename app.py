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
@app.route('/predict_credit', methods=['POST'])
def predict_credit():
    try:
        data = request.json
        
        amount = float(data.get('amount', 0))
        rate = float(data.get('rate', 0))

        # 2. 构造 DataFrame (列名必须保持 'loan_amnt' 以匹配模型训练时的要求)
        input_data = {
            'loan_amnt': [amount],      # 关键修改：把 amount 填入 loan_amnt
            'loan_int_rate': [rate],    # 关键修改：把 rate 填入 loan_int_rate
        }
        df = pd.DataFrame(input_data)
        
        if model:
            try:
                prediction = model.predict(df)[0]
                result = "Approved (with good credit)" if prediction == 1 else "Rejected (high risk)"
            except Exception as e:
                # 捕获模型内部错误（如特征数量不对），返回错误信息而不是崩溃
                return jsonify({"result": f"Model Error: {str(e)}"}), 200
        else:
            result = "Approved" if amount < 10000 else "Manual Review"
        
        return jsonify({"result": result})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"result": f"System Error: {str(e)}"}), 500

if __name__ == '__main__':
    # host='0.0.0.0' 是在 Codespaces 中正常预览的关键
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)