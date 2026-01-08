<template>
  <div class="app-container">
    <header>
      <div class="header-content">
        <h1>📅 待辦事項日曆</h1>
        <div class="controls">
          <div class="view-toggle">
            <button :class="{ active: currentView === 'calendar' }" @click="currentView = 'calendar'">
              <span class="icon">📅</span> 日曆
            </button>
            <button :class="{ active: currentView === 'list' }" @click="currentView = 'list'">
              <span class="icon">📋</span> 看板
            </button>
          </div>
          <button @click="openNewModal" class="btn-primary">
            + 新增事項
          </button>
        </div>
      </div>
    </header>

    <main>
      <div class="content-wrapper">
        <div class="view-content">
          <CalendarView 
            v-if="currentView === 'calendar'"
            @select-date="handleDateSelect" 
            @select-event="handleEventSelect" 
          />
          <TodoList v-else @edit="handleEventSelect" />
        </div>
      </div>
    </main>

    <EventModal 
      :isOpen="isModalOpen" 
      :editData="currentEvent"
      @close="isModalOpen = false"
      @save="handleSave"
      @delete="handleDelete"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import CalendarView from './components/CalendarView.vue'
import TodoList from './components/TodoList.vue' // 實際內容將變更為看板
import EventModal from './components/EventModal.vue'
import { useTodoStore } from './stores/todo'

const store = useTodoStore()
const currentView = ref('list') // 預設改為看板模式以便查看效果
const isModalOpen = ref(false)
const currentEvent = ref(null)

function openNewModal() {
  currentEvent.value = null
  isModalOpen.value = true
}

function handleDateSelect(data) {
  currentEvent.value = {
    start: data.start,
    end: data.end,
    is_all_day: data.allDay
  }
  isModalOpen.value = true
}

function handleEventSelect(todo) {
  currentEvent.value = todo
  isModalOpen.value = true
}

async function handleSave(formData) {
  try {
    if (formData.id) {
      await store.updateTodo(formData.id, formData)
    } else {
      await store.addTodo(formData)
    }
    isModalOpen.value = false
    store.fetchTodos()
  } catch (e) {
    alert('儲存失敗: ' + e.message)
  }
}

async function handleDelete(id) {
  try {
    await store.deleteTodo(id)
    isModalOpen.value = false
    store.fetchTodos()
  } catch (e) {
    alert('刪除失敗: ' + e.message)
  }
}
</script>

<style>
/* Global Reset & Base Styles */
:root {
  --primary-color: #3b82f6;
  --bg-color: #f3f4f6;
  --text-main: #1f2937;
  --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

body {
  margin: 0;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  background-color: var(--bg-color);
  color: var(--text-main);
  -webkit-font-smoothing: antialiased;
}

.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

header {
  background: white;
  padding: 0 24px;
  height: 64px;
  border-bottom: 1px solid #e5e7eb;
  flex-shrink: 0;
}

.header-content {
  max-width: 1400px;
  margin: 0 auto;
  height: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

h1 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: #111827;
  display: flex;
  align-items: center;
  gap: 8px;
}

.controls {
  display: flex;
  gap: 16px;
  align-items: center;
}

/* Buttons & Toggles */
.btn-primary {
  background: var(--primary-color);
  color: white;
  border: none;
  padding: 8px 20px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  font-size: 0.95rem;
  transition: background 0.2s;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

.btn-primary:hover {
  background: #2563eb;
}

.view-toggle {
  display: flex;
  background: #f3f4f6;
  border-radius: 8px;
  padding: 4px;
  gap: 4px;
}

.view-toggle button {
  background: transparent;
  border: none;
  padding: 6px 16px;
  cursor: pointer;
  border-radius: 6px;
  font-size: 0.9rem;
  color: #6b7280;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s;
}

.view-toggle button.active {
  background: white;
  color: var(--text-main);
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
  font-weight: 600;
}

/* Main Content Area */
main {
  flex: 1;
  padding: 24px;
  overflow: hidden;
}

.content-wrapper {
  max-width: 1400px;
  margin: 0 auto;
  height: 100%;
}

.view-content {
  height: 100%;
  border-radius: 12px;
}
</style>
