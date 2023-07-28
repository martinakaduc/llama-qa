import streamlit as st
from model import llm, hfe, qaprompt
from paperqa import Docs, Doc, PromptCollection
from paperqa.types import Text

if 'docs' not in st.session_state:
    st.session_state.docdb = []
    st.session_state.docs = Docs(llm=llm, embeddings=hfe)
    st.session_state.context = ""


def load_docs(docnames, docfiles):
    for i, f in enumerate(docfiles):
        st.session_state.docs.add_file(f, docname=docnames[i], chunk_chars=500)
    print("Documents added")


def answer_query(query):
    answer = st.session_state.docs.query(query)
    print(answer.formatted_answer)
    return answer


def main():
    st.image(["Logo BK.png", "Logo CSE.png"], width=120)
    st.title("Knowledge-based Question Answering System")

    st.subheader("Contexts")
    # Get user input
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    docnames = []
    unprocessed_docnames = []
    unprocessed_docfiles = []
    if len(uploaded_files) == 0:
        for docname in st.session_state.docdb:
            st.session_state.docs.delete(name=docname)
        st.session_state.docdb = []
    else:
        for uploaded_file in uploaded_files:
            docnames.append(uploaded_file.name)
            if uploaded_file.name not in st.session_state.docdb:
                unprocessed_docnames.append(uploaded_file.name)
                unprocessed_docfiles.append(uploaded_file)

        for processed_file in st.session_state.docdb:
            if processed_file not in docnames:
                st.session_state.docs.delete(name=processed_file)
                st.session_state.docdb.remove(processed_file)

    if len(unprocessed_docnames) > 0:
        with st.spinner('Processing documents...'):
            load_docs(unprocessed_docnames, unprocessed_docfiles)
            st.session_state.docdb.extend(unprocessed_docnames)

    use_manual_context = st.checkbox('Use manual context', value=True)
    if use_manual_context:
        context = st.text_area("Enter context", value="Ho Chi Minh City University of Technology is a member of Vietnam National University Ho Chi Minh City.", height=200, max_chars=500)
        if context != st.session_state.context:
            st.session_state.context = ""
            st.session_state.docs.delete(name="Manual Context")

            if context != "":
                with st.spinner('Processing contexts...'):
                    doc = Doc(docname="Manual Context",
                              citation="", dockey="Manual Context")
                    texts = [Text(
                        text=context,
                        name="Manual Context",
                        doc=doc,
                    )]
                    st.session_state.docs.add_texts(
                        texts, doc=doc)
                    st.session_state.context = context
    else:
        st.session_state.context = ""
        st.session_state.docs.delete(name="Manual Context")

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
