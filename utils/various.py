import re

def concept_in_response(response: dict) -> str:
    return response["context"].get("parent_concept", response["context"]["concept"])

def normalize_filename(filename:str) -> str:
    return filename.replace(" ", "_")

def normalize_model_name(name:str) -> str:
    return name.replace("anthropic.", "").replace("meta.", "").replace("-v1:0", "").replace("-32k", "").replace("-16k", "")

