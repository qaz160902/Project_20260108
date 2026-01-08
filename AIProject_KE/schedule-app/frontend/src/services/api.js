import axios from 'axios';

const API_URL = 'http://127.0.0.1:5000/api';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json'
    }
});

export const taskService = {
    getTasks(startDate, endDate) {
        return api.get('/tasks', {
            params: {
                start_date: startDate,
                end_date: endDate
            }
        });
    },
    createTask(taskData) {
        return api.post('/tasks', taskData);
    },
    updateTask(id, taskData) {
        return api.put(`/tasks/${id}`, taskData);
    },
    deleteTask(id) {
        return api.delete(`/tasks/${id}`);
    }
};
