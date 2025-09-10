<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import Editor from '@/components/Editor.vue'
import ChatBot from '@/components/ChatBot.vue'
import Header from '@/components/Header.vue'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5050'
const route = useRoute()
const filePath = ref('')

// Fired when ChatBot emits new doc-path
function handleDocPath(path: string) {
  console.log('Doc path received from ChatBot:', path)
  filePath.value = path
}

onMounted(() => {
  const file = route.query.file as string
  // if (file) {
  //   // 🔹 Load file immediately on mount if coming from UploadView
  //   filePath.value = `${API_BASE_URL}/download/${file}?t=${Date.now()}`
  //   console.log('Loaded file on mount:', filePath.value)
  // }
})
</script>

<template>
  <a-layout class="layout">
    <Header />
    <a-layout-content>
      <a-row>
        <a-col flex="auto">
          <!-- Editor loads the file as soon as filePath is set -->
          <Editor :fileParam="filePath" />
        </a-col>
        <a-col flex="600px">
          <!-- ChatBot can still override filePath later -->
          <ChatBot :initialFileId="route.query.file as string" @doc-path="handleDocPath" />
        </a-col>
      </a-row>
    </a-layout-content>
  </a-layout>
</template>

<style scoped></style>
