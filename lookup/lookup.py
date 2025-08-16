from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import os, sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing_and_embedding.chroma_db_client import ChromaClient

class Lookup:
    def __init__(self, chroma_db_client, model_name="google/flan-t5-small", max_new_tokens=200, device=-1):
        llm_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=max_new_tokens,
            device=device
        )
        self.llm = HuggingFacePipeline(pipeline=llm_pipeline)
        self.chroma_db_client = chroma_db_client

    def get_qa(self,retriever):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

    def generate_reponse(self, user_id : str, query : str, top_k : int = 5, verbose : bool = True):
        retriever = self.chroma_db_client.get_user_retriever(user_id, top_k)
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain.invoke({"query": query})
    
if __name__ == "__main__":
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Initialize persistent vector DB
    chroma_client = ChromaClient(
        collection_name="all_users_docs",
        embedding_model_name=embedding_model
    )

    lookup_client = Lookup(chroma_client)

    print(lookup_client.generate_reponse("user_a", "Who was delighted with Mr. Bingley?"))

