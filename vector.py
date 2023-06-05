import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()
openai_api = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
directory = "txt"
chroma = Chroma(embedding_function=embeddings, persist_directory="db")

def embed_text_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                chroma.add_documents(text, )                
    chroma.persist()

def get_user_prompt():
    return input("Enter your prompt: ")

def main():
    retriever = chroma.as_retriever(search_kwargs={"k": 3})
    retriever.search_kwargs
    while True:
        prompt = get_user_prompt()
        if prompt.lower() in ['quit', 'exit']:
            break
        qa_chain = RetrievalQA.from_chain_type(llm=openai_api, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
        response = qa_chain(prompt)
        print(response['result'])
        for result in response["source_documents"]:
            document_id = result["id"]
            document_name = result["name"]
            similarity_score = result["score"]
            print(f"Document ID: {document_id}, Name: {document_name}, Score: {similarity_score}")

if __name__ == '__main__':
    main()