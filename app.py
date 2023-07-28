import streamlit as st
import requests
import json

url = "https://bahnar.dscilab.site:20007/llama/query"
headers = {
  'Content-Type': 'application/json'
}

if 'context' not in st.session_state:
    st.session_state.context = ""

def answer_query(query):
    payload = json.dumps({
      "context": f"{st.session_state.context}",
      "question": f"{query}"
    })
    response = requests.request("POST", url, headers=headers, data=payload)
    answer = response.json()["answer"]
    print(answer)
    return answer


def main():
    st.image(["Logo BK.png", "Logo CSE.png"], width=120)
    st.title("Knowledge-based Question Answering System")

    st.subheader("Contexts")
    use_manual_context = st.checkbox('Use manual context', value=True)
    if use_manual_context:
        context = st.text_area("Enter context", value="Ho Chi Minh City University of Technology is a member of Vietnam National University Ho Chi Minh City.", height=200, max_chars=500)
        if context != st.session_state.context:
            st.session_state.context = ""

            if context != "":
                with st.spinner('Processing contexts...'):
                    st.session_state.context = context
    else:
        st.session_state.context = ""

    st.subheader("Question")
    question = st.text_input("Enter your question")

    # Display the answer and additional context
    st.subheader('Answer')
    if question != "":
        with st.spinner('Finding answers...'):
            answer = answer_query(question)
    else:
        answer = ""
    st.write(answer)

    st.write("Powered by LLaMA-2")

if __name__ == "__main__":
    main()
