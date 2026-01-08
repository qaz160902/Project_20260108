from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, Task
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允許跨域請求

# 設定 SQLite 資料庫路徑
# 確保 instance 資料夾存在
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "schedule.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# 初始化資料庫
with app.app_context():
    db.create_all()

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    # 支援透過 start_date 和 end_date 篩選 (格式: YYYY-MM-DD)
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    query = Task.query

    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            query = query.filter(Task.due_date >= start_date)
        except ValueError:
            pass # 忽略錯誤的日期格式

    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            # 通常 end_date 要包含當天，所以這裡可以加一天或是視需求調整
            # 這裡假設使用者傳入的是截止當天的 00:00:00，若要包含當天所有時間，前端需傳入隔天或後端處理
            # 簡單起見，這裡直接比對
            query = query.filter(Task.due_date <= end_date)
        except ValueError:
            pass

    # 依照截止日期排序
    tasks = query.order_by(Task.due_date).all()
    return jsonify([task.to_dict() for task in tasks])

@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.json
    try:
        # 解析日期字串 (預期前端傳送 ISO 格式或 YYYY-MM-DD HH:MM:SS)
        # 為了容錯，這裡嘗試解析 ISO 格式
        due_date = datetime.fromisoformat(data['due_date'].replace('Z', '+00:00'))
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid or missing due_date'}), 400

    new_task = Task(
        title=data.get('title'),
        description=data.get('description', ''),
        due_date=due_date,
        completed=False
    )
    db.session.add(new_task)
    db.session.commit()
    return jsonify(new_task.to_dict()), 201

@app.route('/api/tasks/<int:id>', methods=['PUT'])
def update_task(id):
    task = Task.query.get_or_404(id)
    data = request.json

    if 'title' in data:
        task.title = data['title']
    if 'description' in data:
        task.description = data['description']
    if 'completed' in data:
        task.completed = data['completed']
    if 'due_date' in data:
        try:
            task.due_date = datetime.fromisoformat(data['due_date'].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({'error': 'Invalid due_date format'}), 400

    db.session.commit()
    return jsonify(task.to_dict())

@app.route('/api/tasks/<int:id>', methods=['DELETE'])
def delete_task(id):
    task = Task.query.get_or_404(id)
    db.session.delete(task)
    db.session.commit()
    return jsonify({'message': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
