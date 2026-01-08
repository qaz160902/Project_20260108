<template>
  <div v-if="isOpen" class="modal-overlay" @click.self="close">
    <div class="modal-content">
      <div class="modal-header">
        <h2>{{ isEdit ? '📝 編輯行程' : '✨ 新增行程' }}</h2>
        <button class="close-btn" @click="close">×</button>
      </div>
      
      <form @submit.prevent="save" class="modal-form">
        <div class="form-group full-width">
          <label>事項名稱</label>
          <input v-model="form.title" required placeholder="例如: 產品週會" class="input-main" />
        </div>
        
        <div class="form-row">
          <div class="form-group">
            <label>分類標籤</label>
            <select v-model="form.category" class="input-main">
              <option value="一般">一般</option>
              <option value="工作">工作</option>
              <option value="上課">上課</option>
              <option value="開會">開會</option>
              <option value="私人">私人</option>
            </select>
          </div>
          <div class="form-group center-checkbox">
            <label>全天行程</label>
            <div class="toggle-wrapper">
              <input type="checkbox" v-model="form.is_all_day" id="all-day" />
              <label for="all-day" class="toggle-label"></label>
            </div>
          </div>
        </div>

        <div class="form-row date-row">
          <div class="form-group">
            <label>開始時間</label>
            <input 
              type="datetime-local" 
              v-model="form.start_time" 
              :required="!form.is_all_day" 
              class="input-main"
            />
          </div>
          <div class="form-group" v-if="!form.is_all_day">
            <label>結束時間</label>
            <input 
              type="datetime-local" 
              v-model="form.end_time" 
              class="input-main"
            />
          </div>
        </div>

        <div class="form-group full-width">
          <label>詳細備註</label>
          <textarea v-model="form.description" placeholder="輸入行程細節..." class="input-main"></textarea>
        </div>

        <div class="modal-footer">
          <button type="button" v-if="isEdit" @click="remove" class="btn-delete">刪除事項</button>
          <div class="footer-right">
            <button type="button" @click="close" class="btn-secondary">取消</button>
            <button type="submit" class="btn-save">儲存變更</button>
          </div>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, computed } from 'vue'

const props = defineProps({
  isOpen: Boolean,
  editData: Object
})

const emit = defineEmits(['close', 'save', 'delete'])

const form = ref({
  title: '',
  category: '一般',
  is_all_day: false,
  start_time: '',
  end_time: '',
  description: '',
  completed: false
})

const isEdit = computed(() => !!props.editData && !!props.editData.id)

watch(() => props.isOpen, (newVal) => {
  if (newVal) {
    if (props.editData) {
      form.value = {
        ...props.editData,
        start_time: formatDateTime(props.editData.start),
        end_time: formatDateTime(props.editData.end)
      }
    } else {
      form.value = {
        title: '',
        category: '一般',
        is_all_day: false,
        start_time: '',
        end_time: '',
        description: '',
        completed: false
      }
    }
  }
})

function formatDateTime(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset())
  return d.toISOString().slice(0, 16)
}

function close() {
  emit('close')
}

function save() {
  emit('save', form.value)
}

function remove() {
  if (confirm('確定要刪除這個事項嗎？')) {
    emit('delete', props.editData.id)
  }
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(4px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2000;
}

.modal-content {
  background: white;
  border-radius: 16px;
  width: 550px;
  max-width: 95%;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  overflow: hidden;
  animation: modal-in 0.3s ease-out;
}

@keyframes modal-in {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.modal-header {
  padding: 20px 24px;
  border-bottom: 1px solid #f1f5f9;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h2 {
  margin: 0;
  font-size: 1.25rem;
  color: #1e293b;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  color: #94a3b8;
  cursor: pointer;
}

.modal-form {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-row {
  display: flex;
  gap: 16px;
}

.form-group {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-group.full-width {
  width: 100%;
}

.form-group label {
  font-size: 0.875rem;
  font-weight: 600;
  color: #475569;
}

.input-main {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 0.95rem;
  color: #1e293b;
  box-sizing: border-box;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.input-main:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

textarea.input-main {
  height: 100px;
  resize: vertical;
}

/* Date row special handling to prevent overflow */
.date-row {
  flex-wrap: wrap;
}

.date-row .form-group {
  min-width: 200px;
}

.center-checkbox {
  flex: 0 0 100px;
  align-items: center;
  justify-content: center;
}

.modal-footer {
  margin-top: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.footer-right {
  display: flex;
  gap: 12px;
}

button {
  padding: 10px 20px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-secondary {
  background: #f1f5f9;
  border: none;
  color: #475569;
}

.btn-save {
  background: #3b82f6;
  border: none;
  color: white;
}

.btn-delete {
  background: white;
  border: 1px solid #fee2e2;
  color: #ef4444;
}

.btn-delete:hover {
  background: #fef2f2;
}

.btn-save:hover {
  background: #2563eb;
}

/* Custom Checkbox/Toggle Switch Style */
.toggle-wrapper {
  position: relative;
  width: 44px;
  height: 24px;
}

.toggle-wrapper input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-label {
  position: absolute;
  cursor: pointer;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: #e2e8f0;
  transition: .4s;
  border-radius: 24px;
}

.toggle-label:before {
  position: absolute;
  content: "";
  height: 18px; width: 18px;
  left: 3px; bottom: 3px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

input:checked + .toggle-label {
  background-color: #3b82f6;
}

input:checked + .toggle-label:before {
  transform: translateX(20px);
}
</style>