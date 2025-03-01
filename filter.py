from threading import Thread
from df.enhance import init_df, enhance
import torch
import wave
import numpy as np
import queue

class Filter(Thread):
    RATE = 48000  # DeepFilterNet is optimized for 48kHz
    def __init__(self, shared_queue, event):
        super().__init__()
        self.model, self.df_state, _ = init_df("./DeepFilterNet3", config_allow_defaults=True)
        self.model.eval()
        self.collected_audio_data = np.array([], dtype=np.int16)
        self.data_queue = shared_queue
        self.stop_event = event
        self.collected_audio_data

    def run(self):
        while not self.stop_event.is_set():
            try:
                audio_tensor = self.data_queue.get(timeout=3)
                with torch.no_grad():
                    enhanced_tensor = enhance(self.model, self.df_state, 
                                              audio_tensor)
                # Convert tensor back to numpy array
                enhanced_tensor = enhanced_tensor.squeeze(0)
                enhanced_audio_int16 = (enhanced_tensor * (1<<15)).to(torch.int16)
                self.collected_audio_data = np.append(self.collected_audio_data,
                                                 enhanced_audio_int16)
            except queue.Empty:
                self.save()
        print("filter thread stopping and saving")
        self.save()

    def save(self):
        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit PCM (2 bytes per sample)
            wf.setframerate(Filter.RATE)  # Sample rate
            wf.writeframes(self.collected_audio_data.tobytes())  # Convert NumPy array to bytes and write

