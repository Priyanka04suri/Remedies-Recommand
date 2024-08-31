from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
   Please provide detailed information about the disease Disease . The information should include:

   Causes: Explain the underlying causes or risk factors associated with this disease.
   Symptoms: List the common and significant symptoms that patients might experience.
   Diagnosis Methods: Describe the medical tests, exams, or procedures commonly used to diagnose this disease.
   Available Treatments: Outline the standard treatment options, including medications, therapies, or surgical interventions.
   Potential Complications: Discuss any serious or long-term complications that may arise if the disease is left untreated or poorly managed.
   Recommended Lifestyle Changes: Suggest lifestyle modifications that can help prevent or manage this disease effectively.
 
    Context:\n {context}?\n
    Disease: \n{user_disease}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_disease"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and get a response
def user_input(user_disease):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_disease)

    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Generate the response
    response = chain.invoke({"input_documents": docs, "user_disease": user_disease}, return_only_outputs=True)
    print(response)

# Main function to run the script
def main():
    user_disease = "malaria"
    user_input(user_disease)

if __name__ == "__main__":
    main()
