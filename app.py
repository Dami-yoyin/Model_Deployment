# app.py
import streamlit as st

def main():
    st.title("Streamlit Greeting App")
    name = st.text_input("Enter your name", "John Doe")
    greeting = f"Hello, {name}!"
    st.write(greeting)

if __name__ == "__main__":
    main()
