import numpy as np
import librosa
import pyaudiowpatch as pyaudio
import threading
from loguru import logger

CHUNK = 4096
SAMPLE_RATE = 44100
N_MFCC = 13
SIMILARITY_THRESHOLD = 0.82


def load_reference(filepath: str) -> np.ndarray:
    """Load and extract mean MFCCs from the reference .ogg file."""
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).mean(axis=1)


def compare_mfcc(ref_mfcc: np.ndarray, audio_chunk: np.ndarray) -> float:
    """Return cosine similarity between reference MFCCs and a captured chunk."""
    if np.max(np.abs(audio_chunk)) < 1e-4:
        # Silence — skip comparison
        return 0.0
    chunk_mfcc = librosa.feature.mfcc(y=audio_chunk, sr=SAMPLE_RATE, n_mfcc=N_MFCC).mean(axis=1)
    norm = np.linalg.norm(ref_mfcc) * np.linalg.norm(chunk_mfcc)
    if norm == 0:
        return 0.0
    return float(np.dot(ref_mfcc, chunk_mfcc) / norm)


def get_loopback_device(pa: pyaudio.PyAudio):
    """Find the default WASAPI loopback device (system audio output)."""
    try:
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        raise RuntimeError("WASAPI not available on this system.")

    default_output_index = wasapi_info["defaultOutputDevice"]
    default_output = pa.get_device_info_by_index(default_output_index)

    # Find the loopback variant of the default output device
    for i in range(pa.get_device_count()):
        device = pa.get_device_info_by_index(i)
        if device.get("isLoopbackDevice") and device["name"] == default_output["name"]:
            return device

    raise RuntimeError(
        "Could not find a WASAPI loopback device. "
        "Make sure your audio output device supports loopback."
    )


def seek_sound(reference_path: str, stop_event: threading.Event) -> bool:
    """
    Capture system audio via WASAPI loopback and detect when FishBite.ogg plays.

    :param reference_path: Absolute or relative path to FishBite.ogg
    :param stop_event: Set this event externally to cancel detection
    :return: True if the sound was detected, False if stopped early
    """
    ref_mfcc = load_reference(reference_path)
    pa = pyaudio.PyAudio()

    try:
        device = get_loopback_device(pa)
    except RuntimeError as e:
        logger.error(str(e))
        pa.terminate()
        return False

    channels = device["maxInputChannels"] or 1
    rate = int(device["defaultSampleRate"])

    logger.info(
        f"Loopback device: {device['name']} | "
        f"{channels}ch @ {rate}Hz"
    )

    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=CHUNK,
    )

    logger.info("Listening for fish bite sound on system audio...")
    detected = False

    try:
        while not stop_event.is_set():
            raw = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(raw, dtype=np.float32)

            # Mix down to mono if stereo
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            # Resample if the device rate differs from our target
            if rate != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)

            score = compare_mfcc(ref_mfcc, audio)
            logger.debug(f"Sound similarity: {score:.3f}")

            if score >= SIMILARITY_THRESHOLD:
                logger.info("Fish bite sound detected!")
                detected = True
                break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return detected
