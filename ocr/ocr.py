# Ejecutar ocr_form.py para procesar la imagen y generar el archivo JSON correspondiente
   
# -----------------------------
# Importa las bibliotecas necesarias
from collections import namedtuple
import pytesseract
import cv2
import json
import subprocess
import os

# Define las rutas de entrada y salida
input_dir = '/Users/mcc/desarrollo/codigo/datasetgts/images/'  # Carpeta para las imágenes originales
output_dir = '/Users/mcc/desarrollo/codigo/datasetgts/img/'  # Carpeta para las imágenes renombradas
json_output_dir = '/Users/mcc/desarrollo/codigo/datasetgts/json/'  # Carpeta para los archivos JSON generados por OCR

# Define la ruta al script de OCR
ocr_script = "libocr.py"

# Obtiene la lista de archivos en la carpeta de entrada
image_files = sorted(os.listdir(input_dir))

# Recorre los archivos de la carpeta de entrada
for i, filename in enumerate(image_files):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Solo procesa archivos de imagen
        # Obtiene el número de la imagen del índice actual
        img_no = str(i + 1)
        print(img_no)
        # Define las rutas de entrada y salida para esta imagen
        img_path = os.path.join(input_dir, filename)
        new_img_path = os.path.join(output_dir, f"img_{img_no}.png")
        json_output_path = os.path.join(json_output_dir, f"img_{img_no}.json")

        # Lee la imagen en escala de grises
        image = cv2.imread(img_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Guarda la imagen en escala de grises con el nuevo nombre en la carpeta de salida
        cv2.imwrite(new_img_path, gray_image)

        # Ejecuta el script de OCR en la imagen actual
        subprocess.run(["python", ocr_script, "-i", new_img_path, "-o", json_output_path])

