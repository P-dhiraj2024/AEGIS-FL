"""AEGIS-FL (Paper B) code."""
import os

# many experiment processes share one GPU: limit allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
