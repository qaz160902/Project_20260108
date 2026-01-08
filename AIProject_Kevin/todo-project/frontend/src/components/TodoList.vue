<template>
  <div class="kanban-board">
    <div class="column today">
      <div class="column-header">
        <h2>🔥 今天</h2>
        <span class="count">{{ todayTodos.length }}</span>
      </div>
      <div class="task-list">
        <div 
          v-for="todo in todayTodos" 
          :key="todo.id" 
          class="task-card"
          @click="$emit('edit', todo)"
        >
          <div class="card-status-line" :style="{ background: getCategoryColor(todo.category) }"></div>
          <div class="card-content">
            <div class="card-header">
              <span class="category-tag" :style="{ color: getCategoryColor(todo.category), background: getCategoryColor(todo.category) + '20' }">
                {{ todo.category }}
              </span>
              <input 
                type="checkbox" 
                :checked="todo.completed" 
                @click.stop 
                @change="toggleComplete(todo)" 
                class="checkbox"
              />
            </div>
            <h3 :class="{ completed: todo.completed }">{{ todo.title }}</h3>
            <div class="card-footer">
              <span class="time" v-if="todo.start_time">
                ⏰ {{ formatTime(todo.start_time) }}
              </span>
            </div>
          </div>
        </div>
        <div v-if="todayTodos.length === 0" class="empty-state">
          今天沒有行程 🎉
        </div>
      </div>
    </div>

    <div class="column this-week">
      <div class="column-header">
        <h2>📅 這禮拜</h2>
        <span class="count">{{ thisWeekTodos.length }}</span>
      </div>
      <div class="task-list">
        <div 
          v-for="todo in thisWeekTodos" 
          :key="todo.id" 
          class="task-card"
          @click="$emit('edit', todo)"
        >
          <div class="card-status-line" :style="{ background: getCategoryColor(todo.category) }"></div>
          <div class="card-content">
            <div class="card-header">
              <span class="category-tag" :style="{ color: getCategoryColor(todo.category), background: getCategoryColor(todo.category) + '20' }">
                {{ todo.category }}
              </span>
              <input 
                type="checkbox" 
                :checked="todo.completed" 
                @click.stop 
                @change="toggleComplete(todo)" 
                class="checkbox"
              />
            </div>
            <h3 :class="{ completed: todo.completed }">{{ todo.title }}</h3>
            <div class="card-footer">
              <span class="time" v-if="todo.start_time">
                🗓️ {{ formatDayTime(todo.start_time) }}
              </span>
            </div>
          </div>
        </div>
        <div v-if="thisWeekTodos.length === 0" class="empty-state">
          這週其餘時間空閒
        </div>
      </div>
    </div>

    <div class="column next-week">
      <div class="column-header">
        <h2>🚀 下禮拜</h2>
        <span class="count">{{ nextWeekTodos.length }}</span>
      </div>
      <div class="task-list">
        <div 
          v-for="todo in nextWeekTodos" 
          :key="todo.id" 
          class="task-card"
          @click="$emit('edit', todo)"
        >
          <div class="card-status-line" :style="{ background: getCategoryColor(todo.category) }"></div>
          <div class="card-content">
            <div class="card-header">
              <span class="category-tag" :style="{ color: getCategoryColor(todo.category), background: getCategoryColor(todo.category) + '20' }">
                {{ todo.category }}
              </span>
              <input 
                type="checkbox" 
                :checked="todo.completed" 
                @click.stop 
                @change="toggleComplete(todo)" 
                class="checkbox"
              />
            </div>
            <h3 :class="{ completed: todo.completed }">{{ todo.title }}</h3>
            <div class="card-footer">
              <span class="time" v-if="todo.start_time">
                🗓️ {{ formatDate(todo.start_time) }}
              </span>
            </div>
          </div>
        </div>
        <div v-if="nextWeekTodos.length === 0" class="empty-state">
          下週暫無安排
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useTodoStore } from '../stores/todo'
import { computed, onMounted } from 'vue'

const store = useTodoStore()
const emit = defineEmits(['edit'])

onMounted(() => {
  store.fetchTodos()
})

// Date Helpers
const now = new Date()
const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate())
const todayEnd = new Date(todayStart)
todayEnd.setDate(todayEnd.getDate() + 1)

// End of this week (Saturday)
const dayOfWeek = todayStart.getDay() // 0 (Sun) to 6 (Sat)
const endOfThisWeek = new Date(todayStart)
endOfThisWeek.setDate(todayStart.getDate() + (6 - dayOfWeek) + 1) // Next Sunday 00:00

// End of next week
const endOfNextWeek = new Date(endOfThisWeek)
endOfNextWeek.setDate(endOfNextWeek.getDate() + 7)

const todayTodos = computed(() => {
  return store.todos.filter(t => {
    if (!t.start_time) return false // No date = backlog? (ignored for now)
    const d = new Date(t.start_time)
    return d >= todayStart && d < todayEnd
  }).sort((a, b) => new Date(a.start_time) - new Date(b.start_time))
})

const thisWeekTodos = computed(() => {
  return store.todos.filter(t => {
    if (!t.start_time) return false
    const d = new Date(t.start_time)
    return d >= todayEnd && d < endOfThisWeek
  }).sort((a, b) => new Date(a.start_time) - new Date(b.start_time))
})

const nextWeekTodos = computed(() => {
  return store.todos.filter(t => {
    if (!t.start_time) return false
    const d = new Date(t.start_time)
    return d >= endOfThisWeek && d < endOfNextWeek
  }).sort((a, b) => new Date(a.start_time) - new Date(b.start_time))
})

function toggleComplete(todo) {
  store.updateTodo(todo.id, { completed: !todo.completed })
}

function getCategoryColor(category) {
  switch (category) {
    case '工作': return '#3b82f6'
    case '上課': return '#10b981'
    case '開會': return '#f59e0b'
    case '私人': return '#ef4444'
    default: return '#6b7280'
  }
}

function formatTime(isoStr) {
  return new Date(isoStr).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatDayTime(isoStr) {
  const d = new Date(isoStr)
  const dayName = ['日', '一', '二', '三', '四', '五', '六'][d.getDay()]
  return `週${dayName} ${formatTime(isoStr)}`
}

function formatDate(isoStr) {
  const d = new Date(isoStr)
  return `${d.getMonth() + 1}/${d.getDate()} (${['日', '一', '二', '三', '四', '五', '六'][d.getDay()]})`
}
</script>

<style scoped>
.kanban-board {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
  height: 100%;
  overflow-x: auto;
  padding: 2px; /* 給予陰影呼吸空間 */
  padding-bottom: 10px;
}

.column {
  background: #f8fafc; /* Slightly darker than white card */
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  height: 100%;
  border: 1px solid #e2e8f0;
}

.column-header {
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #e2e8f0;
  background: white;
  border-radius: 12px 12px 0 0;
}

.column-header h2 {
  margin: 0;
  font-size: 1rem;
  color: #334155;
  font-weight: 600;
}

.count {
  background: #e2e8f0;
  color: #475569;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 600;
}

.task-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.task-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  border: 1px solid #f1f5f9;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}

.task-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.card-status-line {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
}

.card-content {
  padding: 12px 12px 12px 16px; /* Extra left padding for status line */
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 8px;
}

.category-tag {
  font-size: 0.75rem;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 600;
}

.checkbox {
  width: 18px;
  height: 18px;
  cursor: pointer;
  margin: 0;
}

.task-card h3 {
  margin: 0 0 8px 0;
  font-size: 0.95rem;
  color: #1e293b;
  line-height: 1.4;
}

.task-card h3.completed {
  text-decoration: line-through;
  color: #94a3b8;
}

.card-footer {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  color: #64748b;
}

.empty-state {
  text-align: center;
  color: #94a3b8;
  padding: 20px 0;
  font-size: 0.9rem;
  font-style: italic;
}

/* Scrollbar styling for columns */
.task-list::-webkit-scrollbar {
  width: 6px;
}
.task-list::-webkit-scrollbar-track {
  background: transparent;
}
.task-list::-webkit-scrollbar-thumb {
  background-color: #cbd5e1;
  border-radius: 3px;
}

/* RWD for smaller screens */
@media (max-width: 768px) {
  .kanban-board {
    grid-template-columns: 1fr; /* Stack columns on mobile */
    overflow-y: auto;
  }
  .column {
    height: auto;
    max-height: 500px; /* Limit height per column on mobile */
  }
}
</style>