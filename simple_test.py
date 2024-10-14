# simple_test.py
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.llms import OpenAI
    print("Core langchain_community imports are successful.")
except ModuleNotFoundError as e:
    print(f"Import error: {e}")
except ImportError as e:
    print(f"Import error: {e}")
