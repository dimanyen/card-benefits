
# PDF Document Retrieval-Augmented Generation System / PDF 文件檢索增強生成系統

This project demonstrates how to create a Retrieval-Augmented Generation (RAG) system using Langchain and OpenAI's GPT-4, integrated with PDF data stored in a vector database. The example is designed to run in a Google Colab environment with the necessary dependencies.

本項目展示如何使用 Langchain 和 OpenAI 的 GPT-4 創建一個檢索增強生成 (RAG) 系統，集成了存儲在向量數據庫中的 PDF 數據。該示例設計在 Google Colab 環境中運行，並提供了必要的依賴項。

# Sample
問題: "我想要去日本玩，有麼優惠推薦"
![Sample Image](https://github.com/dimanyen/card-benefits/blob/main/sample.png?raw=true)

## Prerequisites / 先決條件

- Python 3.7+
- Google Colab
- An OpenAI API Key
- Google Drive account

- Python 3.7+
- Google Colab
- OpenAI API 金鑰
- Google Drive 帳戶

## Setup / 設置

1. **Google Drive Mounting / 掛載 Google Drive**:
   - Ensure your Google Drive is mounted in your Google Colab environment to access PDF files and save the vector store.
   - 確保在 Google Colab 環境中掛載了 Google Drive 以訪問 PDF 文件並保存向量存儲。

2. **Dependencies / 依賴項**:
   - Install the required libraries in your Colab environment:
     ```python
     !pip install langchain chromadb
     ```
   - 在 Colab 環境中安裝所需庫：
     ```python
     !pip install langchain chromadb
     ```

## Usage / 使用方法

### Step 1: Import Libraries and Set API Key / 步驟 1：導入庫並設置 API 金鑰

```python
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from google.colab import drive
from langchain.docstore.document import Document
import chromadb
from chromadb.config import Settings
from google.colab import userdata

api_key = userdata.get('OPENAI_API_KEY')
```

### Step 2: Mount Google Drive / 步驟 2：掛載 Google Drive

```python
drive.mount('/content/drive', force_remount=True)
```

### Step 3: Set Folder Path and Initialize PDF Loaders / 步驟 3：設置資料夾路徑並初始化 PDF 加載器

```python
folder_path = "/content/信用卡優惠訊息"

pdf_loaders = []
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        pdf_loaders.append(PyPDFLoader(file_path))

if not pdf_loaders:
    raise FileNotFoundError(f"No PDF files found in directory: {folder_path}")
```

### Step 4: Load PDF Documents / 步驟 4：加載 PDF 文件

```python
documents = []
for loader in pdf_loaders:
    documents.extend(loader.load())

if not documents:
    raise ValueError("No documents were loaded.")
```

### Step 5: Split Texts / 步驟 5：分割文本

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
texts = text_splitter.split_documents(documents)

if not texts:
    raise ValueError("Text splitting failed.")
```

### Step 6: Initialize OpenAI Embeddings / 步驟 6：初始化 OpenAI 嵌入

```python
embeddings = OpenAIEmbeddings(api_key=api_key)
document_texts = [doc.page_content for doc in texts]

if not document_texts:
    raise ValueError("No text content found in documents.")
```

### Step 7: Embed Documents / 步驟 7：嵌入文檔

```python
document_embeddings = embeddings.embed_documents(document_texts)

if not document_embeddings:
    raise ValueError("Document embedding failed.")
```

### Step 8: Initialize Chroma Client / 步驟 8：初始化 Chroma 客戶端

```python
chroma_client = chromadb.Client(Settings(persist_directory="/content/drive/MyDrive/vector_store"))
```

### Step 9: Create Collection and Add Documents / 步驟 9：創建集合並添加文檔

```python
collection = chroma_client.create_collection(name="document_collection")

for i, (embedding, text) in enumerate(zip(document_embeddings, document_texts)):
    collection.add(ids=[str(i)], embeddings=[embedding], metadatas=[{"text": text}])

print("Documents have been successfully added to the Chroma vector store.")
```

### Step 10: Query Similar Documents / 步驟 10：查詢相似文檔

```python
query = "我時要去日本玩，有麼推薦"
query_embedding = embeddings.embed_query(query)
results = collection.query(query_embeddings=[query_embedding], n_results=5)
similar_texts = [metadata["text"] for metadata in results["metadatas"][0]]

display(similar_texts)
```

### Step 11: Generate Responses with OpenAI GPT-4 / 步驟 11：使用 OpenAI GPT-4 生成回應

```python
from openai import OpenAI
from IPython.display import Markdown, display

client = OpenAI(api_key=api_key)
completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "根據我輸入的問題與內容生成結果"},
    {"role": "user", "content": f'問題:{query}
內容:{similar_texts}'}
  ]
)

display(Markdown(completion.choices[0].message.content))
```

## Conclusion / 結論

This project sets up a RAG system using PDF documents stored in Google Drive, processed with Langchain, and stored in a Chroma vector database. OpenAI's GPT-4 is then used to generate responses based on user queries and the retrieved document data.

本項目設置了一個使用存儲在 Google Drive 中的 PDF 文件，通過 Langchain 處理，並存儲在 Chroma 向量數據庫中的 RAG 系統。然後使用 OpenAI 的 GPT-4 根據用戶查詢和檢索的文檔數據生成回應。

## License / 許可證

This project is licensed under the MIT License.

本項目基於 MIT 許可證。
