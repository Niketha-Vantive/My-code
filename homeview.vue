<script setup lang="ts">
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'

const router = useRouter()

function handleStart() {
  try {
    router.push({ name: 'UploadView' })
  } catch (err) {
    console.error(err)
    message.error('Navigation failed.')
  }
}
</script>

<template>
  <div class="page-wrapper">
    <div class="container">
      <img src="../assets/logo.png" alt="Vantive Logo" class="intro-image" />

      <h1 class="title">Welcome to Vantive's Smart Document Assistant</h1>

      <p class="lead">
        Automatically fill out your document templates using intelligent assistance.
      </p>

      <ul class="instructions">
        <li><strong>Step 1:</strong> Click the <strong>Start</strong> button below.</li>
        <li><strong>Step 2:</strong> You’ll be directed to a page featuring a <strong>chat assistant and a document preview section</strong> , where you can choose the type of design document you wish to Autofill.</li>
        <li><strong>Step 3:</strong> Answer a few questions from the AI assistant and see your document getting autofilled.</li>
        <li><strong>Step 4:</strong> Download your fully filled-out document.</li>
      </ul>

      <a-button type="primary" size="large" class="btn-start" @click="handleStart">Start</a-button>
    </div>
  </div>
</template>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

.page-wrapper {
  background: linear-gradient(to bottom right, #f4efff, #e8ddff);
  font-family: 'Inter', sans-serif;
  height: 100vh;
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Container styling exactly like HTML */
.container {
  text-align: center;
  max-width: 720px;
  padding: 40px 30px;
  background-color: #fff;
  border-radius: 18px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
  animation: fadeIn 1s ease-in-out;
}

/* Animation */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Logo */
.intro-image {
  max-height: 270px;
  margin-bottom: 25px;
  object-fit: contain;
}

/* Title styling */
.title {
  font-size: 1.9rem;
  font-weight: 700;
  color: #4b0082;
  margin-bottom: 16px;
}

/* Subtext */
.lead {
  font-size: 1.08rem;
  color: #444;
  margin-bottom: 24px;
}

/* Instructions */
.instructions {
  text-align: left;
  margin-top: 10px;
  margin-bottom: 25px;
  color: #333;
  font-size: 0.98rem;
  list-style-type: none;
  padding-left: 0;
}

.instructions li {
  margin-bottom: 12px;
  line-height: 1.5;
}

/* Start button */
.btn-start {
  font-size: 1.05rem;
  padding: 12px 30px;
  border-radius: 8px;
}

/* Mobile responsive */
@media (max-width: 576px) {
  .title {
    font-size: 1.6rem;
  }
}
</style>
