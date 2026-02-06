import os
from flask import Flask, render_template, request,jsonify
from groq import Groq  # 对应 w8 groq.ipynb 的学习内容 
import pandas as pd
import joblib

app = Flask(__name__)

# 1. 配置 Groq 客户端 (建议在 Codespaces Settings 中设置环境变量)
GROQ_API_KEY = os.environ.get("Project A", "your_default_api_key_here")
client = Groq(api_key="Project A")

# 2. 加载信用评估模型 (对应 W3 练习成果 )
# 注意：请确保将您的模型文件命名为 credit_model.pkl 并放在 models 文件夹下
try:
    model = joblib.load('models/credit_model.pkl')
except:
    model = None
    print("提醒：未找到模型文件，将使用模拟逻辑。")

@app.route('/')
def index():
    return render_template('index.html')

# 模块一：AI 聊天机器人 (基于 Groq )
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192", # 或者您常用的 mixtral-8x7b-32768
            messages=[
                {"role": "system", "content": "你是一位专业的区域银行助手，负责协助客户办理业务。"},
                {"role": "user", "content": user_input}
            ],
        )
        bot_response = completion.choices[0].message.content
        return jsonify({"status": "success", "response": bot_response})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# 模块二：智能贷款审批 (基于 W3 逻辑 )
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
        result = "批准（信用良好）" if prediction == 1 else "拒绝（高风险）"
    else:
        # 兜底模拟逻辑
        result = "批准" if int(data.get('amount', 0)) < 10000 else "需进一步人工审核"
    
    return jsonify({"result": result})

if __name__ == '__main__':
    # host='0.0.0.0' 是在 Codespaces 中正常预览的关键
    app.run(host='0.0.0.0', port=5000, debug=True)