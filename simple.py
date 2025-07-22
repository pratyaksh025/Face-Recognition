import streamlit as st


st.title("Simple Streamlit App")
list_of_operations =["+", "-", "*", "/"]
num1 = st.number_input("Enter first number:", value=0)
selected_operation = st.selectbox("Select an operation:", list_of_operations)
num2 = st.number_input("Enter second number:", value=0)
if st.button("Calculate"):
    result = (lambda x: f"Calculating {num1} {selected_operation} {num2} = {eval(f'{num1}{selected_operation}{num2}')}")(None)
    st.success(f"Result: {result}")

