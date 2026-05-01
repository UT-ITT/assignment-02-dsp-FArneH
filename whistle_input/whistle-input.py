import sounddevice as sd
import numpy as np
from pynput.keyboard import Key, Controller
from collections import deque
import threading
import time

CHUNK_SIZE = 512
RATE       = 44100
CHANNELS   = 1
SILENCE    = 0.010    
WHISTLE_LO = 400      
WHISTLE_HI = 4000     
WINDOW     = 20       
MIN_VALID  = 0.50     
MIN_RANGE  = 80       
SLOPE_THR  = 7        
COOLDOWN   = 25       

keyboard = Controller()
_freq = 0.0
_lock = threading.Lock()


def detect_freq(mono):
    rms = float(np.sqrt(np.mean(mono ** 2)))
    if rms < SILENCE:
        return 0.0
    win  = mono * np.hanning(len(mono))
    mag  = np.abs(np.fft.rfft(win))
    freq = np.fft.rfftfreq(len(mono), 1.0 / RATE)
    lo   = int(np.searchsorted(freq, WHISTLE_LO))
    hi   = int(np.searchsorted(freq, WHISTLE_HI))
    if lo >= hi:
        return 0.0
    return float(freq[np.argmax(mag[lo:hi]) + lo])


def audio_callback(indata, frames, t, status):
    global _freq
    if status:
        print(status)
    with _lock:
        _freq = detect_freq(indata[:, 0])


def get_freq():
    with _lock:
        return _freq


def check_chirp(history):
    valid = [(i, f) for i, f in enumerate(history) if f > 0]
    if len(valid) < WINDOW * MIN_VALID:
        return None
    xs = np.array([v[0] for v in valid], dtype=float)
    ys = np.array([v[1] for v in valid], dtype=float)
    if ys.max() - ys.min() < MIN_RANGE:
        return None
    xm, ym = xs.mean(), ys.mean()
    denom = float(np.sum((xs - xm) ** 2))
    if denom == 0:
        return None
    slope = float(np.sum((xs - xm) * (ys - ym)) / denom)
    if slope > SLOPE_THR:
        return 'up'
    if slope < -SLOPE_THR:
        return 'down'
    return None


print("Available input devices:\n")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"  {i}: {dev['name']}")
input_device = int(input("\nSelect input device: "))
print("\nWhistle up   (ooo→iii)  →  ↑ UP arrow")
print("Whistle down (iii→ooo)  →  ↓ DOWN arrow")

history  = deque(maxlen=WINDOW)
cooldown = 0

stream = sd.InputStream(
    device=input_device,
    channels=CHANNELS,
    samplerate=RATE,
    blocksize=CHUNK_SIZE,
    callback=audio_callback,
    latency='low',
)

with stream:
    while True:
        time.sleep(CHUNK_SIZE / RATE)
        history.append(get_freq())

        if cooldown > 0:
            cooldown -= 1
            continue

        direction = check_chirp(list(history))
        if direction == 'up':
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            print(f"↑  UP   ({get_freq():.0f} Hz)")
            history.clear()
            cooldown = COOLDOWN
        elif direction == 'down':
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            print(f"↓  DOWN ({get_freq():.0f} Hz)")
            history.clear()
            cooldown = COOLDOWN
