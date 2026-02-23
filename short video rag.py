from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.llms import tongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA

# 加载大模型
key = 'sk-6c3498fa32074503b8173f37828b82f3'
llm = tongyi.Tongyi(api_key = key)

# 嵌入模型
embeddings = ZhipuAIEmbeddings(model='embedding-2')

# 加载文本
loader = TextLoader("./txt",encoding='utf-8')
documents = loader.load()

# 分块
text_splitter = RecursiveCharacterTextSplitter(
    chunck_size = 50,
    chunck_overlap = 0,
    separators=["\n\n","\n","。","!","?"]
)
splits = text_splitter.split_documents(documents)

# 初始化向量数据库
db = Chroma.from_documents(
    documents = splits,
    embedding = embeddings,
    persist_directory="./chroma_db3"
)

# 语义搜索
docs = db.similarity_search("迟到处罚",k=1)
for doc in docs:
    print(doc.page_content)
response = llm.invoke("夏天适合吃什么水果")
print("模型回答：", response)

# 创建RAG链
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = db.as_retriever(search_kwargs={"k":1}),
    return_source_documents = True
)

query = "年假有效期"
result = qa_chain({"query":query})
print("答案：",result['result'])
print("来源：",result['source_documents'][0].pagecontent)
