from embed import save_rgb_embeddings
from upload import upload_to_qdrant

def main():
    save_rgb_embeddings()
    upload_to_qdrant()

if __name__ == "__main__":
    main()
