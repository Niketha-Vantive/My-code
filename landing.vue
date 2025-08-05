<script setup lang="ts">
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'

const router = useRouter()

const data: any[] = [
  {
    title: 'Step 1: Click the Start button below.',
  },
  {
    title: "Step 2: You will be prompted to a page with a chatbot and Document preview",
  },
  {
    title: "Step 3: Use the chat assistant to select the design document required'",
  },
  {
    title: 'Step 4: Answer a few questions from the AI assistant.',
  },
  {
    title: 'Step 5: Download your fully filled-out document.',
  },
]

function handleStart() {
  try {
    router.push({ name: 'EditorView' })
  } catch (err) {
    console.error(err)
    message.error('Upload failed.')
  }
}
</script>

<template>
  <a-layout style="min-height: 100vh; min-width: 100vw">
    <a-layout-content class="content-center">
      <a-card :bordered="true" class="upload-card">
        <img src="../assets/logo.png" width="250px" alt="Logo" />
        <h1>Welcome to Vantive's Smart Document Assistant</h1>
        <p>Automatically fill out your document templates using intelligent assistance.</p>
        <a-list item-layout="horizontal" :data-source="data">
          <template #renderItem="{ item }">
            <a-list-item>
              <a-list-item-meta>
                <template #title>
                  <p>{{ item.title }}</p>
                </template>
              </a-list-item-meta>
            </a-list-item>
          </template>
        </a-list>
        <a-button block type="primary" class="mt-2" @click="handleStart()"> Start </a-button>
      </a-card>
    </a-layout-content>
  </a-layout>
</template>

<style scoped>
.content-center {
  display: flex;
  justify-content: center;
  flex-direction: column;
  align-items: center;
  height: 100%;
  padding: 24px;
  background: linear-gradient(to bottom right, #f4efff, #e8ddff);
}

.upload-card {
  width: 100%;
  max-width: 600px;
  animation: fadeIn 1s ease-in-out;
}

/* ✨ Fade-in animation keyframes */
@keyframes fadeIn {
  0% {
    opacity: 0;
    transform: translateY(20px);
  }
  100% {
    opacity: 1;
    transform: translateY(0);
  }
}

.drop-zone {
  position: relative;
  border: 2px dashed #d9d9d9;
  padding: 40px;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 20px;
  text-align: center;
  transition: all 0.3s;
}

.drop-zone.dragging {
  border-color: #1890ff;
  background-color: #e6f7ff;
}

.drop-zone input[type='file'] {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

.preview-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.file-name {
  font-weight: 500;
  color: #333;
  text-align: center;
}

.preview-img {
  max-width: 250px;
  max-height: 300px;
  object-fit: contain;
  border: 1px solid #f0f0f0;
  border-radius: 4px;
}

.nutrient-container {
  width: 100%;
  height: 400px;
  border: 1px solid #f0f0f0;
  border-radius: 4px;
}
</style>
