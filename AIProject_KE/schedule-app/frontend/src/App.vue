<template>
  <div class="container">
    <header>
      <h1>我的日程管理</h1>
      <div class="view-switcher">
        <button 
          :class="{ active: currentView === 'dashboard' }" 
          @click="currentView = 'dashboard'"
        >
          看板視圖
        </button>
        <button 
          :class="{ active: currentView === 'calendar' }" 
          @click="currentView = 'calendar'"
          style="margin-left: 10px;"
        >
          日曆視圖
        </button>
      </div>
    </header>

    <!-- 新增任務表單 (兩個視圖都顯示，或只在看板顯示，這裡保留都顯示) -->
    <section class="add-task-form">
      <h3>新增日程 / 會議</h3>
      <div class="form-group">
        <input v-model="newTask.title" placeholder="任務標題 (例如：與客戶 A 開會)" />
      </div>
      <div class="form-group">
        <textarea v-model="newTask.description" placeholder="詳細內容..."></textarea>
      </div>
      <div class="form-group">
        <input type="datetime-local" v-model="newTask.due_date" />
      </div>
      <button @click="addTask">加入清單</button>
    </section>

    <!-- 看板視圖 -->
    <div v-if="currentView === 'dashboard'" class="dashboard">
      <!-- 今天 -->
      <div class="column">
        <h2>今天 <span>{{ tasksGrouped.today.length }}</span></h2>
        <div v-for="task in tasksGrouped.today" :key="task.id" class="task-card" :class="{ completed: task.completed }">
          <div class="task-time">{{ formatTime(task.due_date) }}</div>
          <div class="task-title">{{ task.title }}</div>
          <div class="task-desc">{{ task.description }}</div>
          <div class="actions">
            <button @click="toggleComplete(task)">{{ task.completed ? '取消' : '完成' }}</button>
            <button @click="deleteTask(task.id)" style="background: #e53e3e; margin-left: 5px;">刪除</button>
          </div>
        </div>
      </div>

      <!-- 本週 -->
      <div class="column">
        <h2>本週 <span>{{ tasksGrouped.thisWeek.length }}</span></h2>
        <div v-for="task in tasksGrouped.thisWeek" :key="task.id" class="task-card" :class="{ completed: task.completed }">
          <div class="task-time">{{ formatDate(task.due_date) }}</div>
          <div class="task-title">{{ task.title }}</div>
          <div class="task-desc">{{ task.description }}</div>
          <div class="actions">
             <button @click="toggleComplete(task)">{{ task.completed ? '取消' : '完成' }}</button>
             <button @click="deleteTask(task.id)" style="background: #e53e3e; margin-left: 5px;">刪除</button>
          </div>
        </div>
      </div>

      <!-- 下週 -->
      <div class="column">
        <h2>下週 <span>{{ tasksGrouped.nextWeek.length }}</span></h2>
        <div v-for="task in tasksGrouped.nextWeek" :key="task.id" class="task-card" :class="{ completed: task.completed }">
          <div class="task-time">{{ formatDate(task.due_date) }}</div>
          <div class="task-title">{{ task.title }}</div>
          <div class="task-desc">{{ task.description }}</div>
          <div class="actions">
             <button @click="toggleComplete(task)">{{ task.completed ? '取消' : '完成' }}</button>
             <button @click="deleteTask(task.id)" style="background: #e53e3e; margin-left: 5px;">刪除</button>
          </div>
        </div>
      </div>
    </div>

    <!-- 日曆視圖 -->
    <CalendarView v-if="currentView === 'calendar'" :tasks="tasks" />

  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue';
import { taskService } from './services/api';
import { isToday, isThisWeek, addWeeks, isAfter, parseISO, format, startOfWeek, endOfWeek } from 'date-fns';
import CalendarView from './components/CalendarView.vue';

const tasks = ref([]);
const currentView = ref('dashboard'); // 'dashboard' or 'calendar'
const newTask = ref({
  title: '',
  description: '',
  due_date: ''
});

const fetchTasks = async () => {
  try {
    const response = await taskService.getTasks();
    tasks.value = response.data;
  } catch (error) {
    console.error('獲取任務失敗', error);
  }
};

const addTask = async () => {
  if (!newTask.value.title || !newTask.value.due_date) {
    alert('請填寫標題與時間');
    return;
  }
  try {
    await taskService.createTask(newTask.value);
    newTask.value = { title: '', description: '', due_date: '' };
    fetchTasks();
  } catch (error) {
    alert('新增失敗');
  }
};

const toggleComplete = async (task) => {
  try {
    await taskService.updateTask(task.id, { completed: !task.completed });
    fetchTasks();
  } catch (error) {
    console.error('更新失敗', error);
  }
};

const deleteTask = async (id) => {
  if (!confirm('確定刪除？')) return;
  try {
    await taskService.deleteTask(id);
    fetchTasks();
  } catch (error) {
    console.error('刪除失敗', error);
  }
};

// 任務分類邏輯
const tasksGrouped = computed(() => {
  const grouped = {
    today: [],
    thisWeek: [],
    nextWeek: []
  };

  const now = new Date();
  const nextWeekStart = startOfWeek(addWeeks(now, 1), { weekStartsOn: 1 });
  const nextWeekEnd = endOfWeek(addWeeks(now, 1), { weekStartsOn: 1 });

  tasks.value.forEach(task => {
    const date = parseISO(task.due_date);
    
    if (isToday(date)) {
      grouped.today.push(task);
    } else if (isThisWeek(date, { weekStartsOn: 1 }) && isAfter(date, now)) {
      grouped.thisWeek.push(task);
    } else if (isAfter(date, nextWeekStart) && !isAfter(date, nextWeekEnd)) {
      grouped.nextWeek.push(task);
    }
  });

  return grouped;
});

const formatTime = (dateStr) => format(parseISO(dateStr), 'HH:mm');
const formatDate = (dateStr) => format(parseISO(dateStr), 'MM/dd HH:mm');

onMounted(fetchTasks);
</script>