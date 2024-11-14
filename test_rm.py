from openai import OpenAI
import numpy as np



def test_embeddings():
    # Initialize the client
    client = OpenAI(
        # base_url="http://10.169.39.24:11961/v1",  # Replace with your vLLM server URL
        base_url="http://0.0.0.0:8001/v1",  # Replace with your vLLM server URL
        api_key="dummy-key",
    )

    # Test texts
    texts = [
        "Hello, world!",
        "This is a test sentence.",
        "I want to see if embeddings work."
    ]

    response = client.embeddings.create(
        model="/mnt/chatbot30TB/junghyunlee/juneyang/Qwen/Qwen2.5-Math-RM-72B",  # Replace with your model name
        input=texts
    )

    print(f"Number of embeddings generated: {len(response.data)}")
    response.data[0].embedding

    try:
        # Get embeddings for the texts
        response = client.embeddings.create(
            model="/mnt/chatbot30TB/junghyunlee/juneyang/Qwen/Qwen2.5-Math-RM-72B",  # Replace with your model name
            input=texts
        )

        # Print basic information about the response
        print(f"Number of embeddings generated: {len(response.data)}")
        print(f"Embedding dimension: {len(response.data[0].embedding)}")

        # Print the first few dimensions of each embedding
        for i, embedding in enumerate(response.data):
            print(f"\nEmbedding {i + 1} (first 5 dimensions):")
            print(f"Text: {texts[i]}")
            print(f"Dimensions: {embedding.embedding[:5]}")

        # Calculate similarity between first two embeddings as a demo
        if len(response.data) >= 2:
            embedding1 = np.array(response.data[0].embedding)
            embedding2 = np.array(response.data[1].embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            print(f"\nCosine similarity between first two texts: {similarity:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_embeddings()