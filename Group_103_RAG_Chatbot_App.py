import streamlit as st

def main():
    st.title("Group 103 Advanced RAG Chatbot")
    st.write("JPMorgan Chase Financial Statements Query System")
    
    query = st.text_input("Enter your question about JPMorgan Chase financials:")
    if st.button("Submit"):
        if guardrail_input(query):
            top_chunks, ranked = advanced_retrieve(query)
            prompt = generate_prompt(query, top_chunks)
            answer = generate_answer(prompt)
            
            st.subheader("Answer")
            st.write(answer)
            
            st.subheader("Confidence Scores for Retrieved Chunks")
            for text, score in ranked:
                st.write(f"Score: {score:.2f} - Excerpt: {text[:150]}...")
        else:
            st.error("Query contains disallowed content.")

if __name__ == "__main__":
    main()
