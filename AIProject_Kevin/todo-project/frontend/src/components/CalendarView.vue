<template>
  <div class="calendar-container">
    <FullCalendar :options="calendarOptions" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import FullCalendar from '@fullcalendar/vue3'
import dayGridPlugin from '@fullcalendar/daygrid'
import timeGridPlugin from '@fullcalendar/timegrid'
import interactionPlugin from '@fullcalendar/interaction'
import { useTodoStore } from '../stores/todo'

const props = defineProps(['filterCategory'])
const emit = defineEmits(['select-date', 'select-event'])
const store = useTodoStore()

const calendarOptions = computed(() => ({
  plugins: [dayGridPlugin, timeGridPlugin, interactionPlugin],
  initialView: 'dayGridMonth',
  headerToolbar: {
    left: 'prev,next today',
    center: 'title',
    right: 'dayGridMonth,timeGridWeek,timeGridDay'
  },
  events: store.calendarEvents,
  editable: true,
  selectable: true,
  selectMirror: true,
  dayMaxEvents: true,
  weekends: true,
  select: handleDateSelect,
  eventClick: handleEventClick,
  eventDrop: handleEventDrop,
  eventResize: handleEventResize // Resize also changes end time
}))

onMounted(() => {
  store.fetchTodos()
})

function handleDateSelect(selectInfo) {
  emit('select-date', {
    start: selectInfo.startStr,
    end: selectInfo.endStr,
    allDay: selectInfo.allDay
  })
}

function handleEventClick(clickInfo) {
  const event = clickInfo.event
  const todoData = {
    id: parseInt(event.id),
    title: event.title,
    start: event.start,
    end: event.end,
    is_all_day: event.allDay,
    ...event.extendedProps
  }
  emit('select-event', todoData)
}

function handleEventDrop(dropInfo) {
  updateEventDate(dropInfo.event)
}

function handleEventResize(resizeInfo) {
  updateEventDate(resizeInfo.event)
}

function updateEventDate(event) {
  store.updateTodo(parseInt(event.id), {
    start_time: event.start.toISOString(),
    end_time: event.end ? event.end.toISOString() : null,
    is_all_day: event.allDay
  })
}
</script>

<style scoped>
.calendar-container {
  height: 100%;
  padding: 20px;
  background: white;
  border-radius: 12px;
  box-shadow: var(--card-shadow, 0 4px 6px -1px rgba(0, 0, 0, 0.1));
  color: #222;
}

/* FullCalendar Customization */
:deep(.fc-header-toolbar) {
  margin-bottom: 24px !important;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px; /* Prevent overlap on small screens */
}

:deep(.fc-toolbar-chunk) {
  display: flex;
  align-items: center;
  gap: 8px;
}

:deep(.fc-toolbar-title) {
  font-size: 1.5rem !important;
  font-weight: 700;
  color: #1f2937;
  min-width: 200px; /* Ensure space for title */
  text-align: center;
  margin: 0 24px !important; /* Add extra breathing room around title */
}

:deep(.fc-button) {
  background-color: white !important;
  border: 1px solid #e5e7eb !important;
  color: #374151 !important;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  padding: 8px 16px !important;
  font-weight: 500;
  text-transform: capitalize;
  transition: all 0.2s;
}

:deep(.fc-button:hover) {
  background-color: #f9fafb !important;
  border-color: #d1d5db !important;
  color: #111827 !important;
}

:deep(.fc-button-active) {
  background-color: #3b82f6 !important;
  border-color: #3b82f6 !important;
  color: white !important;
}

:deep(.fc-button-primary:not(:disabled):active) {
  background-color: #2563eb !important;
  border-color: #2563eb !important;
}

:deep(.fc-button:focus) {
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5) !important;
}
</style>
