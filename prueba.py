import os
import json
import chromadb

client = chromadb.PersistentClient(path="chromafile")
collection = client.get_or_create_collection(name="partes")


results = collection.query(
   query_texts=["con número de cliente 240"],
   n_results=5
   )


#print("\nConjunto 2:")
print(results)