<script setup lang="ts">
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'

const router = useRouter()

const steps: string[] = [
  'Step 1: Click the Start button below.',
  "Step 2: Upload a PDF document you'd like to autofill.",
  'Step 3: Answer a few questions from the AI assistant.',
  'Step 4: Download your fully filled-out document.',
]

function handleStart() {
  try {
    router.push({ name: 'EditorView' })
  } catch (err) {
    console.error(err)
    message.error('Navigation failed.')
  }
}
</script>

<template>
  <div class="page-wrapper">
    <div class="container">
      <!-- Replace the src below if your image path is different -->
      <img src="../assets/intro-image.png" alt="Welcome" class="intro-image" />

      <h1>Welcome to Vantive's Smart Document Assistant</h1>

      <p class="lead">
        Automatically fill out your document templates using intelligent assistance.
      </p>

      <ul class="instructions">
        <li>
          <strong>Step 1:</strong> Click the <strong>Start</strong> button below.
        </li>
        <li>
          <strong>Step 2:</strong> Upload a <strong>PDF</strong> document you'd like to autofill.
        </li>
        <li>
          <strong>Step 3:</strong> Answer a few questions from the AI assistant.
        </li>
        <li>
          <strong>Step 4:</strong> Download your fully filled-out document.
        </li>
      </ul>

      <a-button type="primary" size="large" class="btn-start" @click="handleStart">
        Start
      </a-button>
    </div>
  </div>
</template>

<style scoped>
/* PAGE BACKGROUND */
.page-wrapper {
  background: linear-gradient(to bottom right, #f4efff, #e8ddff);
  font-family: 'Inter', sans-serif;
  height: 100vh;
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* CONTAINER CARD */
.container {
  text-align: center;
  max-width: 720px;
  padding: 40px 30px;
  background-color: #fff;
  border-radius: 18px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
  animation: fadeIn 1s ease-in-out;
}

/* FADE-IN ANIMATION */
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

/* HEADER */
h1 {
  font-size: 2.4rem;
  font-weight: 700;
  color: #4b0082;
  margin-bottom: 20px;
}

/* LEAD TEXT */
.lead {
  font-size: 1.15rem;
  color: #444;
  margin-bottom: 25px;
}

/* IMAGE */
.intro-image {
  max-height: 280px;
  margin-bottom: 25px;
  width: 100%;
  object-fit: contain;
}

/* BUTTON STYLE */
.btn-start {
  font-size: 1.1rem;
  padding: 12px 30px;
  border-radius: 8px;
  margin-top: 25px;
}

/* INSTRUCTION LIST */
.instructions {
  text-align: left;
  margin-top: 25px;
  color: #333;
  font-size: 0.97rem;
  list-style: none;
  padding-left: 0;
}

.instructions li {
  margin-bottom: 12px;
  line-height: 1.5;
}

/* RESPONSIVE DESIGN */
@media (max-width: 576px) {
  h1 {
    font-size: 1.8rem;
  }
}
<link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet" />

</style>
