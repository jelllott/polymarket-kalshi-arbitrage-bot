import numpy as np

def load_audio(path, target_sr=16000):
    import soundfile as sf
    wav, sr = sf.read(path)
    return wav.astype(np.float32)
