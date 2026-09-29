import importlib.metadata
import os
import platform
from dataclasses import dataclass

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    model_name: str
    revision: str
    dtype: str


def choose_dtype():
    if torch.cuda.is_available():
        major_version, _ = torch.cuda.get_device_capability(0)

        if major_version >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


def resolve_revision(model_name, token=None):
    return HfApi(token=token).model_info(model_name).sha


def load_model(model_name):
    token = os.getenv("HF_TOKEN")
    revision = resolve_revision(model_name, token=token)

    dtype = (
        torch.float32
        if model_name == "google/gemma-3-4b-it"
        else choose_dtype()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        token=token,
    )

    model_class = (
        AutoModelForImageTextToText
        if model_name == "google/gemma-3-4b-it"
        else AutoModelForCausalLM
    )

    model = model_class.from_pretrained(
        model_name,
        revision=revision,
        token=token,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=(
            {0: "14GiB", 1: "14GiB", "cpu": "24GiB"}
            if model_name == "google/gemma-3-4b-it"
            else None
        ),
        low_cpu_mem_usage=True,
    )

    model.eval()

    return LoadedModel(
        model,
        tokenizer,
        model_name,
        revision,
        str(dtype).replace("torch.", ""),
    )

def hardware_info():
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
    }


def clear_model(loaded):
    del loaded.model
    del loaded.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
