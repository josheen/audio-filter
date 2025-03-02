from threading import Thread, Condition
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
        self.data_queue : queue.Queue = shared_queue
        self.stop_event = event
        self.condition = Condition()

    def enhance_data(self, audio_tensor):
        with torch.no_grad():
            enhanced_tensor = enhance(self.model, self.df_state,
                                     audio_tensor)
        # Convert tensor back to numpy array
        enhanced_tensor = enhanced_tensor.squeeze(0)
        enhanced_audio_int16 = (enhanced_tensor * (1<<15)).to(torch.int16)
        return enhanced_audio_int16

    def process_input(self):
            audio_tensor = torch.empty(0)
            while not self.data_queue.empty():
                audio_tensor = torch.cat((audio_tensor, self.data_queue.get()), dim=0)
            enhanced_audio_int16 = self.enhance_data(audio_tensor)
            self.collected_audio_data = np.append(self.collected_audio_data, enhanced_audio_int16)

    def run(self):
        while not self.stop_event.is_set():
            with self.condition:
                while self.data_queue.qsize() < 50:
                    self.condition.wait(timeout=3)
                self.process_input()
        print("stopping and saving")
        #handle remaining samples
        if self.data_queue.qsize():
            self.process_input()
        self.save()

    def save(self):
        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit PCM (2 bytes per sample)
            wf.setframerate(Filter.RATE)  # Sample rate
            wf.writeframes(self.collected_audio_data.tobytes())  # Convert NumPy array to bytes and write

