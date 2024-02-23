import streamlit as st
from utils.OAuthClientLib import *
from utils.auth_functions import *

def main():
   
    st.markdown(""" Whether you're a seasoned researcher or a curious learner, our tool is designed to empower 
        your academic journey and unlock new horizons of knowledge. Dive into our vast collection of 
        resources and let the AI do the heavy lifting for you. Here's what you can expect:""")
    
    st.image(image='./images/research-assistant-gpt-01.png', width=600)

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
    st.set_page_config(page_title="Research Assistant GPT", layout="wide", page_icon="🏠")

    # Stylish Header
    st.title("🎓 Research Assistant GPT 🤖")

    # Description Section
    st.subheader("Welcome to the Future of Academic Research!")
    st.caption("""
        Explore the world of academia with ease 🎓! Our AI-powered research assistant 🤖 is here to 
        help you discover and recommend scholarly papers 📚, provide insightful summaries 📄, and 
        guide you through a sea of literature with style and efficiency 🌟.
    """)

    st.divider()

    if "authentication_status" not in st.session_state or st.session_state["authentication_status"] is None:
        google_login_col, login_text_col, github_login_col = st.columns([4,1,4])
        with google_login_col:
            login_google_oauth()

        with github_login_col:
            login_github_oauth()

    # Initialize session state for authentication
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None

    # Sidebar content
    if st.session_state.get("authentication_status"):
        st.sidebar.subheader(f"""Welcome, {st.session_state.get('name', st.session_state["email"])}""")
        if st.sidebar.button("Logout"):
            logout()  
            st.rerun()

    # Main page content
    if st.session_state["authentication_status"]:
        main() 
    else:
        # Authentication (Login/SignUp) options
        menu = ["Login", "SignUp"]
        choice = st.selectbox("**You can also select to Login or SignUp from the below drop down list**", menu)
        if choice == "Login":
            st.markdown(f"**If you don't have an account, please select the sign-up option from the dropdown list to register with your email address.**")
            login()
        elif choice == "SignUp":
            if register_user():
                st.success("Your account has been registered successfully! You can use your email and password to access the app.", icon="✅")

