import pyaudio
import numpy as np
import queue
import torch
from filter import Filter
import threading

# PyAudio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # DeepFilterNet is optimized for 48kHz
CHUNK = 480 # Number of frames per buffer

def main():
    q = queue.Queue()
    stop_event = threading.Event()
    filter = Filter(q, stop_event)
# Initialize PyAudio
    audio = pyaudio.PyAudio()

# Open input stream (microphone)
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("Recording and filtering...")

    try:
        filter.start()
        while True:
            # Read audio data from the microphone
            data = stream.read(CHUNK)
            # Convert byte data to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            # Convert to PyTorch tensor
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            q.put(audio_tensor)

    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping...")

    finally:
        # Close the stream
        filter.join()
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()

