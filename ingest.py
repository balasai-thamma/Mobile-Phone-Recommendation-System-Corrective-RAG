import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer


WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8090
WEAVIATE_GRPC_PORT = 50051

COLLECTION_NAME = "MobilePhones"
CSV_PATH = "Data/Mobiles Dataset.csv"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

client = weaviate.connect_to_local(
    host=WEAVIATE_HOST,
    port=WEAVIATE_HTTP_PORT,
    grpc_port=WEAVIATE_GRPC_PORT
)

try:


    if not client.collections.exists(COLLECTION_NAME):

        client.collections.create(
            name=COLLECTION_NAME,

            vectorizer_config=Configure.Vectorizer.none(),

            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="company", data_type=DataType.TEXT),
                Property(name="model", data_type=DataType.TEXT),
                Property(name="ram", data_type=DataType.TEXT),
                Property(name="processor", data_type=DataType.TEXT),
                Property(name="battery", data_type=DataType.TEXT),
                Property(name="front_camera", data_type=DataType.TEXT),
                Property(name="back_camera", data_type=DataType.TEXT),
                Property(name="price", data_type=DataType.TEXT),
                Property(name="launch_year", data_type=DataType.TEXT),
            ]
        )

        print(f"[ingest] Created collection: {COLLECTION_NAME}")

    else:
        print(f"[ingest] Collection '{COLLECTION_NAME}' already exists — inserting more data into it.")

    collection = client.collections.get(COLLECTION_NAME)

    # ------------------------
    # LOAD DATASET
    # ------------------------

    df = pd.read_csv(CSV_PATH, encoding="latin1")

    print(f"\nTotal Records Found : {len(df)}")

    # ------------------------
    # INGEST DATA (batched)
    # ------------------------

    with collection.batch.fixed_size(batch_size=100) as batch:

        for index, row in df.iterrows():

            chunk_id = f"chunk_{index + 1}"

            chunk = f"""
            Company: {row['Company Name']}
            Model: {row['Model Name']}
            RAM: {row['RAM']}
            Processor: {row['Processor']}
            Battery: {row['Battery Capacity']}
            Front Camera: {row['Front Camera']}
            Back Camera: {row['Back Camera']}
            Price: {row['Launched Price (India)']}
            Launch Year: {row['Launched Year']}
            """

            if index < 5:
                print("\n" + "=" * 60)
                print(f"Chunk ID : {chunk_id}")
                print(chunk)

            embedding = embedding_model.encode(chunk)

            batch.add_object(
                properties={
                    "chunk_id": chunk_id,
                    "company": str(row["Company Name"]),
                    "model": str(row["Model Name"]),
                    "ram": str(row["RAM"]),
                    "processor": str(row["Processor"]),
                    "battery": str(row["Battery Capacity"]),
                    "front_camera": str(row["Front Camera"]),
                    "back_camera": str(row["Back Camera"]),
                    "price": str(row["Launched Price (India)"]),
                    "launch_year": str(row["Launched Year"])
                },
                vector=embedding.tolist()
            )

    if collection.batch.failed_objects:
        print(f"\n⚠ {len(collection.batch.failed_objects)} objects failed to import:")
        for fail in collection.batch.failed_objects[:5]:
            print(fail)
    else:
        print("\nAll records inserted with no failures.")

    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print("Chunking Method : Row Based Chunking")
    print(f"Total Chunks Created : {len(df)}")
    print(f"Embedding Model : {EMBEDDING_MODEL_NAME}")
    print("Vector Database : Weaviate")
    print("=" * 60)

finally:
    client.close()