import os
import json
import chromadb


# Ruta de la carpeta que contiene los archivos JSON
folder_path = "/home/maikel/vscode/gestai/ocr/json"
client = chromadb.PersistentClient(path="chromafile")
collection = client.get_or_create_collection(name="partes")


# Listas para almacenar los elementos únicos de los diccionarios
lista_1 = []
lista_2 = []
lista_3 = []
diccionario_2 = dict()

# Iterar sobre cada archivo en la carpeta
for index, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Verificar si la clave existe antes de acceder a ella
        let_parte = data.get("let_parte", "")
        no_parte = data.get("no_parte", "")
        fecha = data.get("data", "")
        cliente = data.get("cliente", "")
        no_hoja = data.get("no_hoja", "")
        client_desp = data.get("client_desp", "")
        client_prov = data.get("client_prov", "")

        # Si la clave existe, se accede a su valor. Si no, se establece un valor predeterminado (en este caso, una cadena vacía).
        client_addr1 = data.get("client_addr1", "")
        client_addr2 = data.get("client_addr2", "")

        # Crear la etiqueta
        etiquetas = f'El parte de la letra "{let_parte}" con número de parte "{no_parte}" fecha "{fecha}" con número de cliente "{cliente}" con el número de hoja "{no_hoja}" con la descripción "{client_desp}" con la dirección "{client_addr1}, {client_addr2}" en la provincia "{client_prov}" se encuentra ubicado en "{file_path}"'

        # Agregar elementos a las listas
        lista_1.append(etiquetas)
        lista_3.append(str(index))  # Usando el índice como ID único

# Crear un diccionario a partir de la lista_1
arreglo_dics = [{"tipo_doc": "parte"} for _ in lista_1]


# Imprimir conjuntos
#print("Conjunto 1:")
#print(conjunto_1)
#print("\nConjunto 2:")
#print(conjunto_2)
#print("\nConjunto 3:")
#print(conjunto_3)

documents_list = lista_1
metadata_list = arreglo_dics
ids_list = lista_3

#collection.add(
 #   documents= documents_list,
  #  metadatas= metadata_list,
   # ids=ids_list 
#)

#results = collection.query(
 #   query_texts=["LUANDA"],
  #  n_results=5
#)
#results
results=collection.get(
    ids=["1", "5", "7"],
    where={"tipo_doc": "parte"}
)


#print("\nConjunto 2:")
print(results)
