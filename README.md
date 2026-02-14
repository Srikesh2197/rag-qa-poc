 - streamlit run app.py : execute this from project root folder


Using the app:
 - Choose a model in the left sidebar (OpenAI or Ollama)
        - Need to have ollama installed to run it locally
 - Upload documents
        - Click "Rebuild Index" after adding/removing files
 - Ask questions in the main panel
 - The app always shows the answer and the top 3 source chunks used to answer the question
        - This is configurable in config.py file
 - Can ask follow up questions ; remembers context and history
