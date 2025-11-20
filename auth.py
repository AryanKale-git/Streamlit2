import streamlit as st
from database import register_user, login_user, reset_password

def login():
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username")
    email = st.sidebar.text_input("Email (must be a Gmail)")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if login_user(username, email, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["email"] = email
            st.sidebar.success("Logged in successfully!")
        else:
            st.sidebar.error("Invalid username, email, or password.")

def signup():
    st.sidebar.subheader("Sign Up")
    new_user = st.sidebar.text_input("Create Username")
    new_email = st.sidebar.text_input("Enter your Email (@gmail.com only)")
    new_password = st.sidebar.text_input("Create Password", type="password")

    if st.sidebar.button("Register"):
        if not new_email.endswith("@gmail.com"):
            st.sidebar.error("Invalid email. Please use a valid '@gmail.com' email.")
        elif register_user(new_user, new_email, new_password):
            st.sidebar.success("User registered successfully! Please log in.")
        else:
            st.sidebar.error("Username or email already exists.")

def forgot_password():
    st.sidebar.subheader("Forgot Password?")
    email = st.sidebar.text_input("Enter your registered Email (@gmail.com)")
    new_password = st.sidebar.text_input("Enter New Password", type="password")

    if st.sidebar.button("Reset Password"):
        if reset_password(email, new_password):
            st.sidebar.success("Password reset successfully! Please log in.")
        else:
            st.sidebar.error("Email not found!")

def auth_flow():
    """Handles authentication: Login, Signup, and Password Reset."""
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login()
        signup()
        forgot_password()
    else:
        st.sidebar.success(f"Welcome {st.session_state['username']}!")

