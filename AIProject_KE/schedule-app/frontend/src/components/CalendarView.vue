<template>
  <div class="calendar-container">
    <FullCalendar :options="calendarOptions" />
  </div>
</template>

<script setup>
import { ref, watch, defineProps } from 'vue';
import FullCalendar from '@fullcalendar/vue3';
import dayGridPlugin from '@fullcalendar/daygrid';
import interactionPlugin from '@fullcalendar/interaction';
import zhTwLocale from '@fullcalendar/core/locales/zh-tw';

const props = defineProps({
  tasks: {
    type: Array,
    required: true
  }
});

const calendarOptions = ref({
  plugins: [dayGridPlugin, interactionPlugin],
  initialView: 'dayGridMonth',
  locale: zhTwLocale, // 設定為繁體中文
  headerToolbar: {
    left: 'prev,next today',
    center: 'title',
    right: 'dayGridMonth,dayGridWeek'
  },
  events: [],
  eventClick: handleEventClick
});

//當外部傳入的 tasks 改變時，更新日曆上的事件
watch(() => props.tasks, (newTasks) => {
  calendarOptions.value.events = newTasks.map(task => ({
    id: task.id,
    title: task.title,
    start: task.due_date, // FullCalendar 自動解析 ISO 字串
    backgroundColor: task.completed ? '#a0aec0' : '#42b883', // 完成變灰，未完成綠色
    borderColor: task.completed ? '#a0aec0' : '#42b883',
    extendedProps: {
      description: task.description
    }
  }));
}, { immediate: true });

function handleEventClick(info) {
  alert(`任務：${info.event.title}\n描述：${info.event.extendedProps.description || '無'}\n時間：${info.event.start.toLocaleString()}`);
}
</script>

<style scoped>
.calendar-container {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  margin-top: 20px;
}
</style>
