import weaviate

client = weaviate.connect_to_local(
    host="localhost",
    port=8090,
    grpc_port=50051
)

print("Connected Successfully")

client.close()