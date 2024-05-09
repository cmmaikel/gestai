# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from collections import namedtuple
import pytesseract
import cv2
import json
import sys

# -----------------------------
#   FUNCTIONS
# -----------------------------
def cleanup_text(text):
    # Strip out non-ASCII text so we can draw the text on the image using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# Create a named tuple which we can use to create locations of the input document which is going to be OCR
OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])

# Define the locations of each area of the document which are going to be OCR
OCR_LOCATIONS = [
    OCRLocation("let_parte", (72, 345, 105, 75), ["letra"]),
    OCRLocation("no_parte", (192, 345, 156, 75), ["parte"]),
    OCRLocation("data", (378, 345, 200, 75), ["data"]),
    OCRLocation("cliente", (645, 345, 140, 73), ["cliente"]),
    OCRLocation("no_hoja", (917,364,70,43), ["hoja"]),
    OCRLocation("client_desp", (1464, 303, 912, 117), ["clientdesp"]),
    OCRLocation("client_addr1", (1464, 423, 912, 63), ["clientaddr1"]),
    OCRLocation("client_addr2", (1464, 492, 897, 51), ["clientaddr2"]),
    OCRLocation("client_prov", (1464, 549, 435, 66), ["clienprov"]),
    OCRLocation("client_nif", (2109, 543, 270, 69), ["cliennif"]),
    OCRLocation("client_firma", (1476, 3180, 890, 250), ["clienfirma"])
]

# Load the input image
image_path = sys.argv[2]
image = cv2.imread(image_path)

parsingResults = []

# Loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
    # Extract the OCR ROI from the image
    (x, y, w, h) = loc.bbox
    roi = image[y:y + h, x:x + w]

     # OCR the ROI using Tesseract
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    custom_config = r'--oem 1 --psm 6'
    text = pytesseract.image_to_string(rgb, config=custom_config)

    # Break the text into lines and loop over them
    for line in text.split("\n"):
        # If the line is empty, ignore it
        if len(line) == 0:
            continue
        # Convert the line to lowercase and then check to see if the line contains any of the filter keywords
        # (These keywords are part of the *form itself* and should be ignored)
        lower = line.lower()
        count = sum([lower.count(x) for x in loc.filter_keywords])
        # If the count is zero then we know we are *not* examining a text field that is part of the document itself
        # (ex., info, on the field, an example, help text, etc.)
        if count == 0:
            # Update the parsing results dictionary with the OCR'd text if the line is *not* empty
            parsingResults.append((loc, line))

# Initialize a dictionary to store our final OCR results
results = {}

# Loop over the results of parsing the document
for (loc, line) in parsingResults:
    # Grab any existing OCR result for the current ID of the document
    r = results.get(loc.id, None)
    # If the result is None, initialize it using the text and location namedtuple
    # (Converting it to a dictionary as namedtuples are not hashable)
    if r is None:
        results[loc.id] = line
    # Otherwise, there exists an OCR result for the current area of the document, in order to append to the existing line
    else:
        # Unpack the existing OCR result and append the line to the existing text
        existingText = r
        text = "{}\n{}".format(existingText, line)
        # Update the results dictionary
        results[loc.id] = text

# Write the OCR results to a JSON file
output_json_path = sys.argv[4]
with open(output_json_path, "w") as json_file:
    json.dump(results, json_file)
