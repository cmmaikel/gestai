import os
import json

# Ruta de la carpeta que contiene los archivos JSON
folder_path = "/home/maikel/vscode/gestai/ocr/json"

# Lista para almacenar los diccionarios generados
diccionario_1 = {}
diccionario_2 = {}
diccionario_3 = {}

# Iterar sobre cada archivo en la carpeta
for filename in os.listdir(folder_path):
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


        # Crear los diccionarios
        diccionario_1[filename] = etiquetas
        diccionario_2[filename] = {"tipo_doc": "parte"}
        diccionario_3[filename] = {"id": filename}  # Usando el nombre del archivo como ID único

# Imprimir diccionarios
print("Diccionario 1:")
print(diccionario_1)
print("\nDiccionario 2:")
print(diccionario_2)
print("\nDiccionario 3:")
print(diccionario_3)
