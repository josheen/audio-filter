import pyaudio
import torch
import numpy as np
from df.enhance import init_df, enhance, save_audio
import wave

# Initialize DeepFilterNet model
model, df_state, _ = init_df("./DeepFilterNet2", config_allow_defaults=True)
model.eval()

# PyAudio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # DeepFilterNet is optimized for 48kHz
CHUNK = 4096 # Number of frames per buffer

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open input stream (microphone)
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
collected_audio_data = np.array([], dtype=np.int16)
collected_audio_data_direct = torch.empty((1,0))
print("Recording and filtering...")

try:
    while True:
        # Read audio data from the microphone
        data = stream.read(CHUNK)
        # Convert byte data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        # Convert to PyTorch tensor
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        # Apply DeepFilterNet model
        if audio_tensor is not None:
            with torch.no_grad():
                enhanced_tensor = enhance(model, df_state, audio_tensor)
                #collected_audio_data_direct = torch.cat((collected_audio_data_direct, enhanced_tensor),dim=1)
            # Convert tensor back to numpy array
            enhanced_tensor = enhanced_tensor.squeeze(0)
            enhanced_audio_int16 = (enhanced_tensor * (1<<15)).to(torch.int16)
            collected_audio_data = np.append(collected_audio_data, enhanced_audio_int16)

except KeyboardInterrupt:
    print("Stopping...")
    #np.set_printoptions(threshold=np.inf)
    print(len(collected_audio_data))
    print(collected_audio_data)

    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit PCM (2 bytes per sample)
        wf.setframerate(RATE)  # Sample rate
        wf.writeframes(collected_audio_data.tobytes())  # Convert NumPy array to bytes and write

finally:
    # Close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

