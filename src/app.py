import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- Paths ---
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "health_index")

# --- Load FAISS ---
@st.cache_resource
def load_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

# --- Load local Hugging Face model ---
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        min_length=50,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=pipe)

# --- Streamlit UI ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2F4F4F;  /* Light blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="Fitness Exercise Info", page_icon="💪")
st.title("💪 Health Fitness Guide")
st.write("Get personalized answers about fitness, exercise routines, and health tips. Simply ask a question, and our assistant will provide concise guidance using reliable resources.")

# --- Layout: two columns ---
col1, col2 = st.columns([1, 2])  # ratio: image 1 / text 2

# Column 1: Image from URL
with col1:
    image_url = "https://img.freepik.com/free-vector/healthy-lifestyle-diet-fitness-vector-sign-shape-heart-with-multiple-icons-depicting-various-sports-vegetables-cereals-seafood-meat-fruit-sleep-weight-beverages_1284-44073.jpg"  # <-- replace with your URL
    st.image(image_url, use_container_width=True)


# Column 2: Query input and output
with col2:
    query = st.text_input("Enter your question:")

    if query:
        vectorstore = load_vectorstore()

        # Get top 3 relevant documents
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            st.subheader("Answer:")
            st.write("Did not find anything relevant.")
        else:
            # Take only first 4 sentences from each doc for context
            context_sentences = []
            for d in docs:
                sentences = d.page_content.split(". ")
                context_sentences.extend(sentences[:4])
            short_context = ". ".join(context_sentences)

            # Prompt template
            prompt_template = PromptTemplate(
                template=(
                    "You are a helpful fitness assistant. Answer the question concisely in 2-3 sentences "
                    "using the context below. Ignore any irrelevant parts.\n\n"
                    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                ),
                input_variables=["context", "question"],
            )

            # Call the LLM
            llm = load_llm()
            chain = LLMChain(llm=llm, prompt=prompt_template)
            raw_answer = chain.run(question=query, context=short_context)

            # Return first 3–5 sentences
            sentences = raw_answer.split(". ")
            answer = ". ".join(sentences[:5]).strip()
            if not answer:
                answer = "Did not find anything relevant."

            st.subheader("Answer:")
            st.write(answer)
