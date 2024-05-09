import os
import json
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, DonutProcessor
from transformers import TrainingArguments, Trainer
from huggingface_hub import login, HfFolder
from torch.utils.data import DataLoader

# Iniciar sesión en Hugging Face Hub
login()

# Cargar el conjunto de datos
dataset = load_dataset('maikelcm/part_ocr_text', split='train')

new_special_tokens = [] # new tokens which will be added to the tokenizer
task_start_token = "<s>"  # start of task token
eos_token = "</s>" # eos token of tokenizer

def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def preprocess_documents_for_donut(sample):
    # create Donut-style input
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text) + eos_token
    # convert all images to RGB
    image = sample["image"].convert('RGB')
    return {"image": image, "text": d_doc}

proc_dataset = dataset.map(preprocess_documents_for_donut)

#print(f"Sample: {proc_dataset[45]['text']}")
#print(f"New special tokens: {new_special_tokens + [task_start_token] + [eos_token]}")

# Load processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", use_fast=True)

# add new special tokens to tokenizer
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})

# we update some settings which differ from pretraining; namely the size of the images + no rotation required
# resizing the image to smaller sizes from [1920, 2560] to [960,1280]
processor.feature_extractor.size = [720,960] # should be (width, height)
processor.feature_extractor.do_align_long_axis = False

def transform_and_tokenize(sample, processor=processor, split="train", max_length=512, ignore_id=-100):
    # Definir el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create tensor from image
    try:
        with torch.no_grad():
            # Convertir la imagen a tensor y moverla a la GPU si está disponible
            pixel_values = processor(
                sample["image"].convert("RGB"),  # Convertir la imagen a RGB antes de procesarla
                random_padding=split == "train",
                return_tensors="pt"
            ).pixel_values.squeeze().to(device)
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    # tokenize document
    with torch.no_grad():
        # Procesar el texto en la CPU, ya que no puede ser procesado directamente en la GPU
        input_ids = processor.tokenizer(
            sample["text"],
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token

    # Liberar memoria GPU si es posible
    torch.cuda.empty_cache()

    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}




# DataLoader con almacenamiento en disco y carga perezosa (lazy)
def lazy_load_dataset(split, batch_size, num_workers):
    return DataLoader(
        proc_dataset[split],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Cargar en la CPU y mover a la GPU según sea necesario
        persistent_workers=True  # Reduce el tiempo de carga de los trabajadores
    )

# Procesar el conjunto de datos
processed_dataset = proc_dataset.map(transform_and_tokenize,remove_columns=["image","text"])
processed_dataset = processed_dataset.train_test_split(test_size=0.1)

# Load model from huggingface.co
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# Resize embedding layer to match vocabulary size
new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
print(f"New embedding size: {new_emb}")
# Adjust our image size and output sequence lengths
model.config.encoder.image_size = processor.feature_extractor.size[::-1] # (height, width)
model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))

# Add task token for decoder to start
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]

# Mover el modelo y los datos a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
processed_dataset = processed_dataset.map(lambda x: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()})

# Hyperparameters used for multiple args
hf_repository_id = "maikelcm/donut-ft-cordv2-gestai"

# Arguments for training
training_args = TrainingArguments(
    output_dir=hf_repository_id,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=hf_repository_id,
    hub_token=HfFolder.get_token(),
    fp16=True  # Utilizar FP16 para reducir el uso de memoria y acelerar el entrenamiento
)

# Crear Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    compute_metrics=compute_metrics,
)

# Iniciar entrenamiento
trainer.train()

# Guardar procesador y crear tarjeta de modelo
processor.save_pretrained(hf_repository_id)
trainer.create_model_card()
trainer.push_to_hub()
