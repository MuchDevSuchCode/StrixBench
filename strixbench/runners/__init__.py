"""Engine runners. Each runner turns a model+config spec into BenchResult records."""

from .llamacpp import LlamaCppRunner
from .ollama import OllamaRunner

RUNNERS = {
    "llama.cpp": LlamaCppRunner,
    "ollama": OllamaRunner,
}
