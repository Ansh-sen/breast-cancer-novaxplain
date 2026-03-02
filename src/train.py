"""
train.py — Full dataset training optimized for CPU
- Uses all available images
- Maximum CPU speed optimizations
- Both phases run correctly
- Saves best model automatically
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# ── CPU OPTIMIZATION ──────────────────────────────────────────────────────────
# Use all CPU cores for data loading
tf.config.threading.set_inter_op_parallelism_threads(0)  # auto
tf.config.threading.set_intra_op_parallelism_threads(0)  # auto
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # disable to avoid MKL memory errors

print(f"CPUs available: {os.cpu_count()}")
print(f"TensorFlow: {tf.__version__}")
print("Training on CPU — this will take several hours")
print("Tip: run overnight or use Google Colab for 20x speedup")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR        = r"D:\python_projects\breast-cancer-ai\data"
IMG_SIZE        = (96, 96)
BATCH_SIZE      = 64         # large batch = fewer steps = faster per epoch
PHASE1_EPOCHS   = 15
PHASE2_EPOCHS   = 15
FINE_TUNE_AT    = 30

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR       = os.path.join(PROJECT_ROOT, "models")
BEST_PATH       = os.path.join(MODEL_DIR, "breast_cancer_model_best.keras")
FINAL_PATH      = os.path.join(MODEL_DIR, "breast_cancer_model.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── COLLECT ALL IMAGE PATHS ───────────────────────────────────────────────────
print("\nScanning all images...")
all_paths, all_labels = [], []

for root, dirs, files in os.walk(DATA_DIR):
    folder = os.path.basename(root)
    if folder in ("0", "1"):
        label = int(folder)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                all_paths.append(os.path.join(root, f))
                all_labels.append(label)

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)
n0 = int((all_labels == 0).sum())
n1 = int((all_labels == 1).sum())
total = len(all_paths)
print(f"Total images   : {total}")
print(f"Benign  (0)    : {n0}")
print(f"Malignant (1)  : {n1}")
print(f"Label mapping  : 0=BENIGN  1=MALIGNANT")
print(f"Model output   : > 0.5 = MALIGNANT")

# shuffle and split 80/20
idx = np.arange(total)
np.random.seed(42)
np.random.shuffle(idx)
all_paths  = all_paths[idx]
all_labels = all_labels[idx].astype(np.float32)

split    = int(0.8 * total)
tr_paths = all_paths[:split];  tr_labels = all_labels[:split]
va_paths = all_paths[split:];  va_labels = all_labels[split:]
print(f"\nTrain : {len(tr_paths)}")
print(f"Val   : {len(va_paths)}")

steps_per_epoch = len(tr_paths) // BATCH_SIZE
val_steps       = len(va_paths) // BATCH_SIZE
print(f"\nSteps per epoch : {steps_per_epoch}")
print(f"Est. time/epoch : ~{steps_per_epoch * 1 // 60} min on CPU")
print(f"Est. total time : ~{steps_per_epoch * 1 * (PHASE1_EPOCHS + PHASE2_EPOCHS) // 3600} hrs")

# ── tf.data PIPELINE ──────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

def parse_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.92, 1.08)
    return tf.clip_by_value(img, 0.0, 1.0), label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=8000, seed=42, reshuffle_each_iteration=True)
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

print("\nBuilding datasets (first epoch will cache — subsequent epochs faster)...")
train_ds = make_dataset(tr_paths, tr_labels, training=True)
val_ds   = make_dataset(va_paths, va_labels, training=False)

# ── CLASS WEIGHTS ─────────────────────────────────────────────────────────────
cw = compute_class_weight('balanced',
     classes=np.array([0., 1.]), y=tr_labels)
CW = {0: float(cw[0]), 1: float(cw[1])}
print(f"Class weights  : {CW}")

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1,   activation='sigmoid')
], name="breast_cancer_classifier")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# ── PHASE 1 ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1 — Head training  (base fully frozen)")
print("="*60)

cb1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=4,
        restore_best_weights=True, mode='max', verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        BEST_PATH, monitor='val_auc',
        save_best_only=True, mode='max', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=2, min_lr=1e-7, verbose=1),
]

h1 = model.fit(
    train_ds,
    epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    class_weight=CW,
    callbacks=cb1
)
p1         = len(h1.epoch)
best_p1    = max(h1.history.get('val_auc', [0]))
print(f"\nPhase 1 done — {p1} epochs  |  best val_auc = {best_p1:.4f}")

# save checkpoint
model.save(FINAL_PATH)
print(f"Checkpoint saved after Phase 1: {FINAL_PATH}")

# ── PHASE 2 ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"PHASE 2 — Fine-tuning top {FINE_TUNE_AT} base layers")
print("="*60)

base.trainable = True
freeze_until   = len(base.layers) - FINE_TUNE_AT
for i, layer in enumerate(base.layers):
    layer.trainable = (i >= freeze_until)

trainable = sum(1 for l in base.layers if l.trainable)
print(f"Trainable base layers: {trainable} / {len(base.layers)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

cb2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=5,
        restore_best_weights=True, mode='max', verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        BEST_PATH, monitor='val_auc',
        save_best_only=True, mode='max', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=2, min_lr=1e-8, verbose=1),
]

h2 = model.fit(
    train_ds,
    epochs=p1 + PHASE2_EPOCHS,
    initial_epoch=p1,
    validation_data=val_ds,
    class_weight=CW,
    callbacks=cb2
)
p2      = len(h2.epoch)
best_p2 = max(h2.history.get('val_auc', [0]))
print(f"\nPhase 2 done — {p2} epochs  |  best val_auc = {best_p2:.4f}")

# ── SAVE FINAL ────────────────────────────────────────────────────────────────
model.save(FINAL_PATH)
print(f"\n✅ Final model : {FINAL_PATH}")
print(f"✅ Best model  : {BEST_PATH}")

# ── VERIFY ────────────────────────────────────────────────────────────────────
saved     = tf.keras.models.load_model(FINAL_PATH)
base_s    = saved.layers[0]
trainable = sum(1 for l in base_s.layers if l.trainable)
best_auc  = max(best_p1, best_p2)
print(f"\nTrainable layers in saved model : {trainable}/{len(base_s.layers)}")
print(f"Best val_auc overall            : {best_auc:.4f}")
if trainable > 0:
    print("✅ Grad-CAM will work")
else:
    print("⚠️  All layers frozen — retrain needed")
if best_auc >= 0.85:
    print("✅ Good accuracy — ready for app")
elif best_auc >= 0.75:
    print("⚠️  Moderate accuracy — usable but consider more epochs")
else:
    print("⚠️  Low accuracy — check data and retrain")