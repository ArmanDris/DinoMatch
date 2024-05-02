from embed import save_as_embeddings
from upload import upload_to_qdrant

def main():
    save_as_embeddings()
    upload_to_qdrant()

if __name__ == "__main__":
    main()