import streamlit as st
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from llama_index.schema import NodeWithScore
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List, Optional
import qdrant_client
from qdrant_client import models
import logging

logging.basicConfig(level=logging.INFO)

def load_llm(model_name):
    llm = CTransformers(
        model=model_name,
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0
    )
    return llm

def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    content = ''
    for page in data:
        content += page.page_content
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(content)
    documents = [Document(page_content=t) for t in chunks]
    return documents

class VectorDBRetriever(BaseRetriever):
    """Retriever over a Qdrant vector store."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def llm_pipeline(file_path, model_name):
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Setup Qdrant client and vector store
    client = qdrant_client.QdrantClient(
        "https://5321c90b-0709-43e9-8081-e899e9bc9e94.us-east4-0.gcp.cloud.qdrant.io",
        api_key="eDt8YdkRqa-hdEigsI1C6yV6XFXdThHBcdIcsNoHrUFTqWCByrmn4g",
    )

    client.recreate_collection(
        collection_name="my_collection",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.EUCLID)
    )

    qvs = QdrantVectorStore(client=client, collection_name="my_collection")

    if file_path:
        documents = file_processing(file_path)
        vector_embeddings = [embeddings.embed(document.page_content) for document in documents]
        qvs.add(vector_embeddings)

    retriever = VectorDBRetriever(vector_store=qvs, embed_model=embeddings)

    llm_answer_gen = load_llm(model_name)
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                          chain_type="stuff", 
                                                          retriever=retriever)
    return answer_generation_chain

def run_app():
    st.title("Question over PDF using HF")

    model_selection = st.selectbox(
        'Select a Model:',
        ('TheBloke/Mistral-7B-Instruct-v0.1-GGUF', 'TheBloke/zephyr-7B-alpha-GGUF')
    )

    uploaded_file = st.file_uploader("Upload your PDF file here", type=['pdf'])

    file_path = None
    if uploaded_file:
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        file_path = "temp_pdf.pdf"

    with st.spinner("Analyzing..."):
        answer_generation_chain = llm_pipeline(file_path, model_selection)

    st.success("Analysis complete! You can now ask questions.")

    question = st.text_input("Posez votre question ici")

    if st.button("Ask"):
        with st.spinner("Fetching answer..."):
            query_bundle = QueryBundle(query=question)
            response = answer_generation_chain.run(query_bundle)
            st.write(str(response))

run_app()
