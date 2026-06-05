"""Engine runners. Each runner turns a model+config spec into BenchResult records."""

from .llamacpp import LlamaCppRunner

RUNNERS = {
    "llama.cpp": LlamaCppRunner,
}
