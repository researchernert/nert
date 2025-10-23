# models/embedding_cache.py
"""
Singleton encoder cache for efficient model reuse across the application.
Ensures SentenceTransformer is loaded only ONCE and reused by all components.
"""

from sentence_transformers import SentenceTransformer
from threading import Lock
import os


class EncoderSingleton:
    """
    Thread-safe singleton for SentenceTransformer encoder.
    """
    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not EncoderSingleton._initialized:
            with EncoderSingleton._lock:
                if not EncoderSingleton._initialized:
                    self._initialize()
                    EncoderSingleton._initialized = True

    def _initialize(self):
        """Load model ONCE, reuse forever"""
        print("Loading SentenceTransformer (ONE TIME ONLY)...")

        cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME', None)

        self.encoder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            cache_folder=cache_folder
        )
        print("Encoder loaded and cached globally")

    def encode(self, texts, **kwargs):
        """
        Delegate to the singleton encoder.

        Args:
            texts: String or list of strings to encode
            **kwargs: Additional arguments passed to encoder.encode()
                - batch_size: Number of texts to process at once (default: 32)
                - show_progress_bar: Show progress for large batches (default: False)
                - convert_to_numpy: Return numpy arrays instead of tensors (default: True)

        Returns:
            Embeddings as numpy arrays or tensors
        """
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32
        if 'show_progress_bar' not in kwargs:
            kwargs['show_progress_bar'] = False
        if 'convert_to_numpy' not in kwargs:
            kwargs['convert_to_numpy'] = True

        return self.encoder.encode(texts, **kwargs)

    def get_sentence_embedding_dimension(self):
        """Get embedding dimension (384 for MiniLM)"""
        return self.encoder.get_sentence_embedding_dimension()


_encoder_singleton = None
_init_lock = Lock()


def get_encoder() -> EncoderSingleton:
    """
    Get the shared encoder instance.
    Thread-safe lazy initialization.

    Returns:
        EncoderSingleton: The global encoder instance

    Example:
        encoder = get_encoder()
        embeddings = encoder.encode(['text1', 'text2'], batch_size=32)
    """
    global _encoder_singleton

    if _encoder_singleton is None:
        with _init_lock:
            if _encoder_singleton is None:
                _encoder_singleton = EncoderSingleton()

    return _encoder_singleton


def clear_encoder_cache():
    global _encoder_singleton
    with _init_lock:
        _encoder_singleton = None
        EncoderSingleton._instance = None
        EncoderSingleton._initialized = False
