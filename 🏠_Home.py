import streamlit as st

def main():
    st.set_page_config(page_title="Research Assistant GPT", layout="wide")

    # Stylish Header
    st.title("🎓 Research Assistant GPT 🤖")

    # Description Section
    st.header("Welcome to the Future of Academic Research!")
    st.markdown("""
        Explore the world of academia with ease 🎓! Our AI-powered research assistant 🤖 is here to 
        help you discover and recommend scholarly papers 📚, provide insightful summaries 📄, and 
        guide you through a sea of literature with style and efficiency 🌟.
        
        Whether you're a seasoned researcher or a curious learner, our tool is designed to empower 
        your academic journey and unlock new horizons of knowledge. Dive into our vast collection of 
        resources and let the AI do the heavy lifting for you. Here's what you can expect:
    """)

    st.image(image='./images/research-assistant-gpt-01.png')

    # Key Features Section
    st.subheader("Key Features")
    st.markdown("""
        - 📖 **Vast Knowledge Base**: Access an extensive database of academic papers and articles.
        - 🤔 **Intuitive Question Answering**: Just ask any academic question and get instant answers.
        - 📑 **Summary Generation**: Get quick summaries of lengthy documents to save time.
        - 🔍 **Advanced Search Capabilities**: Easily find papers, articles, and journals relevant to your topic.
        - 📈 **Trend Analysis**: Stay updated with the latest research trends and developments.
        - 📁 **PDF Uploads**: Directly upload PDFs and integrate them into your research.
        - 🎙️ **Voice Queries**: Speak your questions and get spoken answers for an interactive experience.
    """)

    # Call to Action
    st.subheader("Get Started Now!")
    st.markdown("""
        Ready to revolutionize your research experience? Jump right in and start exploring! 
        If you have any PDFs you'd like to include in your search, simply upload them and our 
        AI assistant will incorporate their content into your knowledge base. 
    """)

    st.markdown("---")
    st.markdown("Research Assistant GPT © 2024. All Rights Reserved.")

if __name__ == "__main__":
    main()
