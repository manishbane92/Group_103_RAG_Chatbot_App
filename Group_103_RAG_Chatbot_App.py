import streamlit as st
#from Group_103_RAG_Chatbot_Server import guardrail_input, ask_with_guardrail

def main():
    st.title("Group 103 Advanced RAG Chatbot")
    st.write("JPMorgan Chase Financial Statements Query System")
    
    user_query = st.text_input("Enter your question about JPMorgan Chase financials:")
    #if st.button("Submit"):
    #    if guardrail_input(user_query):
    #        final_answer, top_chunks, cross_ranked = ask_with_guardrail(user_query)
    #        st.subheader("Answer")
    #        st.write(final_answer)
            
    #        st.subheader("Top Retrieved Excerpts and Cross-Encoder Scores")
    #        for text, score in cross_ranked:
    #            st.write(f"Score: {score:.2f} - Excerpt: {text[:150]}...")
    #    else:
    #        st.error("Query contains disallowed content.")

if __name__ == "__main__":
    main()
