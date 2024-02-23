import streamlit as st
import utils.database as db
import re

def is_email_valid(email):
    # Simple regex for validating an email
    pattern = r"^\S+@\S+\.\S+$"
    return re.match(pattern, email)

def is_password_strong(password):
    # Check if the password is at least 6 characters
    if len(password) < 6:
        return False
    return True

def is_email_unique(email):
    # Check if the email is already in the database
    return not db.is_user_exist(email)

def passwords_match(password, repeated_password):
    # Check if both passwords match
    return password == repeated_password

def register_user():
    with st.form("Register User Form"):
        st.subheader("Register User")
        email = st.text_input("Email*")
        name = st.text_input("Name*")
        password = st.text_input("Password*", type="password")
        repeated_password = st.text_input("Repeat Password*", type="password")
        submit_button = st.form_submit_button("Register")

        if submit_button:
            if not is_email_valid(email):
                st.error("Please enter a valid email address.")
                return False
            if not name or name == "":
                st.error("Please enter your name.")
                return False
            if not is_password_strong(password):
                st.error("Password must be at least 6 characters long.")
                return False
            if not passwords_match(password, repeated_password):
                st.error("Passwords do not match.")
                return False
            if not is_email_unique(email):
                st.error("An account with this email already exists.")
                return False
            
            db.insert_user(email, name, password)
            return True
    return False


def login():
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if username and password:
                result, name, email = db.login_user(username, password)
                if result:
                    st.session_state["authentication_status"] = True
                    st.session_state["name"] = name
                    st.session_state["email"] = email
                    st.rerun()
                else:
                    # st.session_state["authentication_status"] = False
                    st.error("Incorrect username or password")
            else:
                st.error("Please enter both username and password")

def logout():
    st.session_state["authentication_status"] = None
    st.session_state["name"] = None
    st.session_state["email"] = None
    st.rerun()