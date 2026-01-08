import { defineStore } from 'pinia'
import axios from 'axios'

const API_URL = 'http://localhost:5001/api/todos'

export const useTodoStore = defineStore('todo', {
  state: () => ({
    todos: [],
    loading: false,
    error: null,
  }),
  getters: {
    calendarEvents: (state) => {
      return state.todos.map(todo => ({
        id: todo.id,
        title: todo.title,
        start: todo.start_time,
        end: todo.end_time,
        allDay: todo.is_all_day,
        extendedProps: {
          description: todo.description,
          category: todo.category,
          completed: todo.completed
        },
        backgroundColor: getCategoryColor(todo.category),
        borderColor: getCategoryColor(todo.category)
      }))
    }
  },
  actions: {
    async fetchTodos(start = null, end = null) {
      this.loading = true
      try {
        const params = {}
        if (start) params.start = start
        if (end) params.end = end
        
        const response = await axios.get(API_URL, { params })
        this.todos = response.data
      } catch (err) {
        this.error = err.message
        console.error('Error fetching todos:', err)
      } finally {
        this.loading = false
      }
    },
    async addTodo(todo) {
      try {
        const response = await axios.post(API_URL, todo)
        this.todos.push(response.data)
        return response.data
      } catch (err) {
        this.error = err.message
        throw err
      }
    },
    async updateTodo(id, updates) {
      try {
        const response = await axios.put(`${API_URL}/${id}`, updates)
        const index = this.todos.findIndex(t => t.id === id)
        if (index !== -1) {
          this.todos[index] = response.data
        }
      } catch (err) {
        this.error = err.message
        throw err
      }
    },
    async deleteTodo(id) {
      try {
        await axios.delete(`${API_URL}/${id}`)
        this.todos = this.todos.filter(t => t.id !== id)
      } catch (err) {
        this.error = err.message
        throw err
      }
    }
  }
})

function getCategoryColor(category) {
  switch (category) {
    case '工作': return '#3788d8' // Blue
    case '上課': return '#10b981' // Green
    case '開會': return '#f59e0b' // Orange
    case '私人': return '#ef4444' // Red
    default: return '#6b7280' // Gray
  }
}
