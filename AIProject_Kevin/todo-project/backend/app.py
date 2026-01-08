from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, Todo
from datetime import datetime
import os

app = Flask(__name__)
CORS(app) # 允許跨來源資源共用 (前端 Vue 存取用)

# 資料庫設定
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# 建立資料表
with app.app_context():
    db.create_all()

@app.route('/api/todos', methods=['GET'])
def get_todos():
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    
    query = Todo.query
    
    # 如果有提供時間區間，進行篩選
    if start_str and end_str:
        try:
            start_dt = datetime.fromisoformat(start_str.replace('Z', ''))
            end_dt = datetime.fromisoformat(end_str.replace('Z', ''))
            query = query.filter(Todo.start_time >= start_dt, Todo.start_time <= end_dt)
        except ValueError:
            return jsonify({"error": "Invalid date format"}), 400
            
    todos = query.all()
    return jsonify([todo.to_dict() for todo in todos])

@app.route('/api/todos', methods=['POST'])
def add_todo():
    data = request.json
    try:
        new_todo = Todo(
            title=data.get('title'),
            description=data.get('description'),
            category=data.get('category', '一般'),
            start_time=datetime.fromisoformat(data['start_time'].replace('Z', '')) if data.get('start_time') else None,
            end_time=datetime.fromisoformat(data['end_time'].replace('Z', '')) if data.get('end_time') else None,
            is_all_day=data.get('is_all_day', False),
            completed=data.get('completed', False)
        )
        db.session.add(new_todo)
        db.session.commit()
        return jsonify(new_todo.to_dict()), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/todos/<int:todo_id>', methods=['PUT'])
def update_todo(todo_id):
    todo = Todo.query.get_or_404(todo_id)
    data = request.json
    
    if 'title' in data: todo.title = data['title']
    if 'description' in data: todo.description = data['description']
    if 'category' in data: todo.category = data['category']
    if 'completed' in data: todo.completed = data['completed']
    if 'start_time' in data: 
        todo.start_time = datetime.fromisoformat(data['start_time'].replace('Z', '')) if data['start_time'] else None
    if 'end_time' in data:
        todo.end_time = datetime.fromisoformat(data['end_time'].replace('Z', '')) if data['end_time'] else None
    if 'is_all_day' in data: todo.is_all_day = data['is_all_day']

    db.session.commit()
    return jsonify(todo.to_dict())

@app.route('/api/todos/<int:todo_id>', methods=['DELETE'])
def delete_todo(todo_id):
    todo = Todo.query.get_or_404(todo_id)
    db.session.delete(todo)
    db.session.commit()
    return jsonify({"message": "Deleted successfully"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
