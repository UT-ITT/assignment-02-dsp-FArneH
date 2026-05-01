import sounddevice as sd
import numpy as np
import pyglet
from pyglet import shapes
import pyglet.text as pt
import threading

CHUNK_SIZE = 1024
RATE       = 44100
CHANNELS   = 1
SILENCE    = 0.010

_freq = 0.0
_rms  = 0.0
_lock = threading.Lock()

def detect_freq(mono):
    rms = float(np.sqrt(np.mean(mono ** 2)))
    if rms < SILENCE:
        return 0.0, rms
    win  = mono * np.hanning(len(mono))
    mag  = np.abs(np.fft.rfft(win))
    freq = np.fft.rfftfreq(len(mono), 1.0 / RATE)
    lo   = int(np.searchsorted(freq, 80))
    hi   = int(np.searchsorted(freq, 1400))
    if lo >= hi:
        return 0.0, rms
    return float(freq[np.argmax(mag[lo:hi]) + lo]), rms

def audio_callback(indata, frames, t, status):
    if status:
        print(status)
    f, r = detect_freq(indata[:, 0])
    with _lock:
        global _freq, _rms
        _freq, _rms = f, r

def get_audio():
    with _lock:
        return _freq, _rms

print("Available input devices:\n")
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        print(f"{i}: {dev['name']}")
input_device = int(input("\nSelect input device: "))

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Ode to Joy – 30 notes (midi note, duration in seconds)
SONG = [
    (64,1.2),(64,1.2),(65,1.2),(67,1.2),
    (67,1.2),(65,1.2),(64,1.2),(62,1.2),
    (60,1.2),(60,1.2),(62,1.2),(64,1.2),
    (64,1.8),(62,0.6),(62,2.4),
    (64,1.2),(64,1.2),(65,1.2),(67,1.2),
    (67,1.2),(65,1.2),(64,1.2),(62,1.2),
    (60,1.2),(60,1.2),(62,1.2),(64,1.2),
    (62,1.8),(60,0.6),(60,2.4),
]

def midi_to_name(n):
    return NOTE_NAMES[n % 12] + str(n // 12 - 1)

# map midi note to normalised screen height
NOTE_LO, NOTE_HI = 59.0, 68.0

def note_to_y(midi):
    return 0.12 + (midi - NOTE_LO) / (NOTE_HI - NOTE_LO) * 0.76

def freq_to_y(f):
    if f <= 0:
        return None
    midi = 69.0 + 12.0 * np.log2(f / 440.0)
    return max(0.04, min(0.96, note_to_y(midi)))

WIN_W, WIN_H = 820, 520
BIRD_X_N   = 0.18
BIRD_R     = 13
GAP_HALF   = 0.14
PIPE_W_N   = 0.07
PIPE_SPEED = 0.007
TOLERANCE  = 1.5   # semitones counted as correct
PIPE_EVERY = 140   # frames between pipes

STATE    = 'intro'
bird_y_n = 0.5
score    = 0
pipes    = []
song_idx = 0
pipe_timer = 0

def reset_game():
    global STATE, bird_y_n, score, pipes, song_idx, pipe_timer
    STATE      = 'playing'
    bird_y_n   = 0.5
    score      = 0
    pipes      = []
    song_idx   = 0
    pipe_timer = 0
    msg_lbl.text   = ""
    sub_lbl.text   = ""
    score_lbl.text = f"Score: 0 / {len(SONG)}"

window = pyglet.window.Window(WIN_W, WIN_H, "Karaoke Game")
batch  = pyglet.graphics.Batch()

grp_bg  = pyglet.graphics.Group(order=0)
grp_mid = pyglet.graphics.Group(order=1)
grp_top = pyglet.graphics.Group(order=2)

shapes.Rectangle(0, 0, WIN_W, WIN_H, color=(15, 20, 40), batch=batch, group=grp_bg)

bird_shape = shapes.Circle(int(BIRD_X_N * WIN_W), int(0.5 * WIN_H), BIRD_R,
                            color=(255, 220, 0), batch=batch, group=grp_top)

PIPE_COLOR  = (30, 200, 30)
pipe_shapes = []
pipe_labels = []
for _ in range(6):
    tr  = shapes.Rectangle(0, 0, 1, 1, color=PIPE_COLOR, batch=batch, group=grp_mid)
    br  = shapes.Rectangle(0, 0, 1, 1, color=PIPE_COLOR, batch=batch, group=grp_mid)
    lbl = pt.Label("", font_name='Arial', font_size=17,
                   color=(255, 255, 255, 255),
                   anchor_x='center', anchor_y='center',
                   batch=batch, group=grp_top)
    tr.visible = br.visible = False
    pipe_shapes.append((tr, br))
    pipe_labels.append(lbl)

score_lbl = pt.Label("Score: 0", font_name='Arial', font_size=16,
    color=(255, 255, 255, 255),
    x=12, y=WIN_H - 24, anchor_x='left', anchor_y='center',
    batch=batch, group=grp_top)

detected_lbl = pt.Label("", font_name='Arial', font_size=13,
    color=(160, 220, 255, 255),
    x=WIN_W // 2, y=18, anchor_x='center', anchor_y='center',
    batch=batch, group=grp_top)

msg_lbl = pt.Label("", font_name='Arial', font_size=20,
    color=(255, 240, 50, 255),
    x=WIN_W // 2, y=WIN_H // 2 + 40,
    anchor_x='center', anchor_y='center',
    batch=batch, group=grp_top)

sub_lbl = pt.Label("", font_name='Arial', font_size=13,
    color=(200, 200, 200, 255),
    x=WIN_W // 2, y=WIN_H // 2 - 10,
    anchor_x='center', anchor_y='center',
    batch=batch, group=grp_top)

def update(dt):
    global STATE, bird_y_n, score, pipes, song_idx, pipe_timer

    freq, rms = get_audio()
    target_y  = freq_to_y(freq)

    if freq > 0:
        detected_lbl.text = f"You: {midi_to_name(round(69.0 + 12.0 * np.log2(freq / 440.0)))}  ({freq:.0f} Hz)"
    else:
        detected_lbl.text = "Sing!"

    if STATE == 'intro':
        msg_lbl.text = "Karaoke Game"
        sub_lbl.text = "Sing a note to start – your pitch controls the bird!"
        if freq > 0:
            reset_game()
        return

    if STATE == 'over':
        if freq > 0:
            reset_game()
        return

    # move bird toward sung pitch
    if target_y is not None:
        bird_y_n += (target_y - bird_y_n) * 0.12
    else:
        bird_y_n += (0.5 - bird_y_n) * 0.02

    bird_y_n = max(0.03, min(0.97, bird_y_n))
    bird_shape.y = int(bird_y_n * WIN_H)

    # spawn next pipe from song
    pipe_timer += 1
    if pipe_timer >= PIPE_EVERY and song_idx < len(SONG):
        pipe_timer = 0
        note, _ = SONG[song_idx]
        song_idx += 1
        pipes.append({'x': 1.06, 'target_note': note, 'scored': False})

    for p in pipes:
        p['x'] -= PIPE_SPEED
    pipes = [p for p in pipes if p['x'] > -PIPE_W_N - 0.05]

    # render pipes
    pw = int(PIPE_W_N * WIN_W)
    for i, ((tr, br), lbl) in enumerate(zip(pipe_shapes, pipe_labels)):
        if i < len(pipes):
            p          = pipes[i]
            gap_cy     = note_to_y(p['target_note'])
            px         = int((p['x'] - PIPE_W_N / 2) * WIN_W)
            gap_top_px = int((gap_cy + GAP_HALF) * WIN_H)
            gap_bot_px = int((gap_cy - GAP_HALF) * WIN_H)
            tr.x = px;  tr.y = gap_top_px
            tr.width = pw;  tr.height = max(1, WIN_H - gap_top_px)
            br.x = px;  br.y = 0
            br.width = pw;  br.height = max(1, gap_bot_px)
            tr.visible = br.visible = True
            lbl.text = midi_to_name(p['target_note'])
            lbl.x    = int(p['x'] * WIN_W)
            lbl.y    = int(gap_cy * WIN_H)
        else:
            tr.visible = br.visible = False
            lbl.text = ""

    # collision and scoring
    for p in pipes:
        gap_cy = note_to_y(p['target_note'])
        if abs(p['x'] - BIRD_X_N) < (PIPE_W_N / 2 + 0.018):
            if not (gap_cy - GAP_HALF < bird_y_n < gap_cy + GAP_HALF):
                _game_over()
                return
        if not p['scored'] and p['x'] + PIPE_W_N / 2 < BIRD_X_N:
            p['scored'] = True
            if freq > 0:
                midi_sung = 69.0 + 12.0 * np.log2(freq / 440.0)
                if abs(midi_sung - p['target_note']) <= TOLERANCE:
                    score += 1

    if song_idx >= len(SONG) and not pipes:
        STATE = 'over'
        pct = int(100 * score / len(SONG))
        msg_lbl.text = f"Song done!  {score}/{len(SONG)} notes  ({pct}%)"
        sub_lbl.text = "Sing any note to play again"
        return

    score_lbl.text = f"Score: {score} / {len(SONG)}"


def _game_over():
    global STATE
    STATE = 'over'
    msg_lbl.text = f"GAME OVER!  Score: {score}/{len(SONG)}"
    sub_lbl.text = "Sing any note to restart"


@window.event
def on_draw():
    window.clear()
    batch.draw()


pyglet.clock.schedule_interval(update, 1 / 60)

stream = sd.InputStream(
    device=input_device,
    channels=CHANNELS,
    samplerate=RATE,
    blocksize=CHUNK_SIZE,
    callback=audio_callback,
    latency='low',
)

with stream:
    pyglet.app.run()
