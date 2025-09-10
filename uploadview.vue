<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { message } from 'ant-design-vue'
import { InboxOutlined, DeleteOutlined, MessageOutlined } from '@ant-design/icons-vue'
import Header from '@/components/Header.vue'
import { useRouter } from 'vue-router'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5050'

type Row = { id: number; name: string; created_at?: string }

const router = useRouter()
const searchText = ref('')
const pagination = ref({ current: 1, pageSize: 10, total: 0 })
const loading = ref(false)

const allFiles = ref<Row[]>([])
const files = ref<Row[]>([])

const isUploadModalVisible = ref(false)
const uploadFileList = ref<any[]>([]) // control a-upload-dragger list

const columns = [
  { title: 'File Name', dataIndex: 'name', key: 'name' },
  { title: 'Created At', dataIndex: 'created_at', key: 'created_at' },
  { title: 'Action', key: 'action', slots: { customRender: 'action' } },
]

function refreshTable() {
  let filtered = allFiles.value
  const q = searchText.value.trim().toLowerCase()
  if (q) filtered = filtered.filter(f => f.name.toLowerCase().includes(q))

  pagination.value.total = filtered.length
  const start = (pagination.value.current - 1) * pagination.value.pageSize
  const end = start + pagination.value.pageSize
  files.value = filtered.slice(start, end)
}

async function handleDelete(record: Row) {
  try {
    const res = await fetch(`${API_BASE_URL}/deleteTemplate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: record.id }),
    })
    if (!res.ok) throw new Error('Failed to delete')
    await res.json()
    message.success(`${record.name} deleted`)

    allFiles.value = allFiles.value.filter(f => f.id !== record.id)
    const maxPage = Math.max(1, Math.ceil(allFiles.value.length / pagination.value.pageSize))
    if (pagination.value.current > maxPage) pagination.value.current = maxPage
    refreshTable()
  } catch (err) {
    console.error(err)
    message.error('Failed to delete file')
  }
}

function handleChat(record: Row) {
  router.push({ path: '/editor', query: { file: record.id } })
}

function handleTableChange(pag: any) {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  refreshTable()
}

function handleUploadClick() {
  isUploadModalVisible.value = true
}

async function handleUploadOk() {
  try {
    if (uploadFileList.value.length === 0) {
      isUploadModalVisible.value = false
      return
    }

    const fileName = uploadFileList.value[0].name
    // 🔹 Check duplicate before upload
    if (allFiles.value.some(f => f.name.toLowerCase() === fileName.toLowerCase())) {
      message.error(`File "${fileName}" already exists!`)
      return
    }

    const formData = new FormData()
    formData.append('file', uploadFileList.value[0].originFileObj)

    const res = await fetch(`${API_BASE_URL}/uploadTemplate`, {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) throw new Error('Upload failed')
    await res.json()
    message.success('Successfully uploaded')
    fetchTemplateList()
  } catch (err) {
    console.error(err)
    message.error('Upload failed')
  } finally {
    isUploadModalVisible.value = false
    uploadFileList.value = []
  }
}

function handleUploadCancel() {
  isUploadModalVisible.value = false
  uploadFileList.value = []
}

function beforeUpload(file: any) {
  const duplicate = allFiles.value.some(f => f.name.toLowerCase() === file.name.toLowerCase())
  if (duplicate) {
    message.error(`File "${file.name}" already exists!`)
    return false // block adding to upload list
  }
  uploadFileList.value = [file]
  return false // prevent auto-upload
}

// 🔹 Fetch template list from API
async function fetchTemplateList() {
  try {
    loading.value = true
    const res = await fetch(`${API_BASE_URL}/templateList`, { method: 'GET' })
    if (!res.ok) throw new Error('Failed to fetch template list')
    const data = await res.json()

    if (Array.isArray(data)) {
      // ✅ backend returned a list
      allFiles.value = data.map(f => ({
        ...f,
        created_at: f.created_at ? new Date(f.created_at).toLocaleString() : '',
      }))
    } else if (Array.isArray(data.templateList)) {
      // ✅ backend returned { templateList: [...] }
      allFiles.value = data.templateList.map(f => ({
        ...f,
        created_at: f.created_at ? new Date(f.created_at).toLocaleString() : '',
      }))
    } else {
      allFiles.value = []
    }
    refreshTable()
  } catch (err) {
    console.error(err)
    message.error('Failed to fetch template list, loading dummy data')
    // dummy fallback with 120 files
    allFiles.value = Array.from({ length: 120 }, (_, i) => ({
      id: i + 1,
      name: `Dummy File ${i + 1}.pdf`,
      created_at: new Date(Date.now() - i * 3600 * 1000).toLocaleString(),
    }))
    refreshTable()
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchTemplateList()
})

// 🔹 Hotload search
watch(searchText, () => {
  pagination.value.current = 1
  refreshTable()
})
</script>

<template>
  <a-layout style="min-height: 100vh; min-width: 100vw">
    <!-- Sticky Main Header -->
    <a-layout-header class="main-header">
      <Header />
    </a-layout-header>

    <a-layout-content class="content-center">
      <a-card title="Templates" :bordered="true" class="upload-card">
        <!-- Search + Upload Button -->
        <div class="table-header">
          <a-input-search
            v-model:value="searchText"
            placeholder="Search files"
            enter-button
            style="max-width: 300px"
          />
          <a-button type="primary" @click="handleUploadClick">
            Upload New Template
          </a-button>
        </div>

        <!-- Files Table -->
        <a-table
          :columns="columns"
          :data-source="files"
          :pagination="pagination"
          :loading="loading"
          row-key="id"
          @change="handleTableChange"
        >
          <template #action="{ record }">
            <a-space>
              <!-- <a-button type="text" danger shape="circle" @click="handleDelete(record)">
                <DeleteOutlined />
              </a-button> -->
              <a-button type="text" shape="circle" @click="handleChat(record)">
                <MessageOutlined />
              </a-button>
            </a-space>
          </template>
        </a-table>
      </a-card>

      <!-- Upload Modal -->
      <a-modal
        v-model:open="isUploadModalVisible"
        title="Upload File"
        @ok="handleUploadOk"
        @cancel="handleUploadCancel"
        ok-text="Done"
      >
        <a-upload-dragger
          name="file"
          :multiple="false"
          v-model:fileList="uploadFileList"
          :before-upload="beforeUpload"
          accept=".pdf,image/*,.doc,.docx"
          style="width: 100%"
        >
          <p class="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p class="ant-upload-text">
            Click or drag file to this area to upload
          </p>
          <p class="ant-upload-hint">
            Support for single file upload.
          </p>
        </a-upload-dragger>
      </a-modal>
    </a-layout-content>
  </a-layout>
</template>

<style scoped>
.main-header {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: #fff;
  padding: 0;
}
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
  max-width: 800px;
  animation: fadeIn 1s ease-in-out;
}
.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
</style>
