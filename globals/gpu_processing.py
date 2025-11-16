import os
import tensorflow as tf
from tensorflow.keras import mixed_precision

def configure_gpu(memory_growth=True, enable_xla=True, use_mixed_precision=True):
    if memory_growth:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs: {gpus}")
        for g in gpus:
            print(f"Configuring memory growth for GPU: {g}")
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

    # Optional: enable XLA JIT
    if enable_xla:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    # Mixed precision speeds up on modern GPUs; keep final Dense in float32
    if use_mixed_precision and tf.config.list_physical_devices('GPU'):
        try:
            mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            # Policy might have been set already; ignore
            pass

def get_tf_strategy(auto_multi_gpu=True):
    """
    Returns a distribution strategy:
      - MirroredStrategy if multiple GPUs and auto_multi_gpu=True
      - Default (no distribution) otherwise
    """
    if auto_multi_gpu and len(tf.config.list_logical_devices('GPU')) > 1:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()