import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO


def _get_size(model):
    try:
        s = model.layers[0].input_shape
        return (int(s[1]), int(s[2]))
    except Exception:
        return (96, 96)


def load_and_preprocess_image(uploaded_file, target_size=None, model=None):
    if target_size is None and model is not None:
        target_size = _get_size(model)
    if target_size is None:
        target_size = (96, 96)
    pil_img = Image.open(uploaded_file).convert("RGB")
    resized  = pil_img.resize(target_size)
    arr      = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0), pil_img


def get_gradcam_heatmap(img_array, model,
                        last_conv_layer_name="block_16_project"):
    base_orig   = model.layers[0]
    h, w        = _get_size(model)
    input_shape = (h, w, 3)

    try:
        base_orig.get_layer(last_conv_layer_name)
        target_name = last_conv_layer_name
    except ValueError:
        convs = [l for l in base_orig.layers
                 if isinstance(l, tf.keras.layers.Conv2D)]
        target_name = convs[-1].name
        print(f"[GradCAM] fallback layer: {target_name}")

    # fresh fully-trainable clone
    base_clone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )
    base_clone.set_weights(base_orig.get_weights())

    target_layer = base_clone.get_layer(target_name)
    extractor    = tf.keras.Model(
        inputs=base_clone.input,
        outputs=[target_layer.output, base_clone.output]
    )

    img_t = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_out, base_out = extractor(img_t, training=True)
        tape.watch(conv_out)
        x = base_out
        for layer in model.layers[1:]:
            x = layer(x, training=False)
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_out)

    if grads is None:
        print("[GradCAM] grads = None")
        sh = conv_out.shape
        return np.zeros((int(sh[1]), int(sh[2])), dtype=np.float32)

    gmax = float(tf.reduce_max(tf.abs(grads)))
    print(f"[GradCAM] gradient max = {gmax:.10f}")

    if gmax < 1e-10:
        sh = conv_out.shape
        return np.zeros((int(sh[1]), int(sh[2])), dtype=np.float32)

    # normalize gradients before pooling
    grads_norm = grads / (gmax + 1e-10)
    pooled     = tf.reduce_mean(grads_norm, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(
        conv_out[0] * pooled[tf.newaxis, tf.newaxis, :], axis=-1
    ).numpy()

    heatmap = np.maximum(heatmap, 0)
    print(f"[GradCAM] after ReLU  max={heatmap.max():.8f}  std={heatmap.std():.8f}")

    if heatmap.max() < 1e-8:
        print("[GradCAM] trying without ReLU...")
        heatmap = tf.reduce_sum(
            conv_out[0] * pooled[tf.newaxis, tf.newaxis, :], axis=-1
        ).numpy()
        heatmap = heatmap - heatmap.min()
        print(f"[GradCAM] no-ReLU max={heatmap.max():.8f}")

    hmax = heatmap.max()
    if hmax < 1e-10:
        sh = conv_out.shape
        return np.zeros((int(sh[1]), int(sh[2])), dtype=np.float32)

    heatmap = (heatmap / hmax).astype(np.float32)
    heatmap  = np.power(heatmap, 0.5)
    print(f"[GradCAM] final std = {heatmap.std():.4f}")
    return heatmap


def create_gradcam_figure(heatmap, pil_img, label, confidence,
                          uncertain=False, target_size=None):
    """
    Create improved Grad-CAM visualization figure with three panels.
    
    Panel 1: Original image
    Panel 2: Prediction label with confidence
    Panel 3: JET heatmap overlay showing important regions
    """
    if target_size is None:
        target_size = (96, 96)
    W, H    = target_size
    orig    = cv2.resize(np.array(pil_img.convert("RGB")), (W, H))
    hm_disp = cv2.resize(heatmap, (W, H))
    
    # Normalize and apply ReLU to heatmap
    hm_norm = np.maximum(hm_disp, 0)
    hm_max = hm_norm.max()
    if hm_max > 1e-10:
        hm_norm = hm_norm / hm_max
    else:
        hm_norm = np.zeros_like(hm_norm)
    
    # Apply JET colormap with proper normalization
    hm_uint8 = np.uint8(255 * hm_norm)
    jet_map = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    jet_map = cv2.cvtColor(jet_map, cv2.COLOR_BGR2RGB)
    
    # Create overlay with 40% transparency
    overlay_img = cv2.addWeighted(
        orig.astype(np.float32), 0.6,
        jet_map.astype(np.float32), 0.4,
        0
    ).astype(np.uint8)
    
    # Find peak activation location for ROI circle
    py, px  = np.unravel_index(np.argmax(hm_norm), hm_norm.shape)
    cr      = max(W, H) // 8
    lc      = "#f43f5e" if label == "MALIGNANT" else "#10b981"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0b0f1a")

    # Panel 1: Original
    axes[0].imshow(orig)
    axes[0].set_title("Original Image", color="white",
                      fontsize=11, fontweight="bold", pad=8)

    # Panel 2: Prediction
    axes[1].imshow(orig)
    axes[1].set_title(f"Predicted: {label}" + ("  \u26a0\ufe0f" if uncertain else ""),
                      color=lc, fontsize=11, fontweight="bold", pad=8)
    axes[1].text(0.5, 0.04, f"{confidence*100:.1f}% confidence",
                 transform=axes[1].transAxes, ha="center", va="bottom",
                 color=lc, fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                           alpha=0.8, edgecolor=lc))

    # Panel 3: JET Heatmap with overlay
    axes[2].imshow(overlay_img)
    axes[2].set_title("Grad-CAM Heatmap (JET: Blue→Green→Yellow→Red)", color="white",
                      fontsize=11, fontweight="bold", pad=8)
    # Mark peak activation region with circle
    circ_color = "yellow" if label == "MALIGNANT" else "cyan"
    axes[2].add_patch(patches.Circle((px, py), radius=cr, linewidth=3,
                      edgecolor=circ_color, facecolor="none", zorder=5, linestyle='--'))

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0b0f1a")

    plt.tight_layout(pad=0.6)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def create_improved_gradcam_heatmap(heatmap, original_size=None):
    """
    Improve Grad-CAM heatmap visualization:
    - Normalize values correctly (0-1 range)
    - Apply ReLU activation  
    - Use JET colormap
    - Proper alpha blending
    
    Returns: Normalized heatmap [0, 255] uint8 for JET colormap
    """
    # Ensure heatmap is float32
    hm = heatmap.astype(np.float32)
    
    # Step 1: Apply ReLU (keep only positive activations)
    hm = np.maximum(hm, 0)
    
    # Step 2: Normalize to [0, 1]
    hm_max = hm.max()
    if hm_max > 1e-10:
        hm = hm / hm_max
    else:
        # If heatmap is empty, return zeros
        return np.zeros_like(hm, dtype=np.uint8)
    
    # Step 3: Convert to uint8 for cv2.applyColorMap
    hm_uint8 = np.uint8(255 * hm)
    
    # Step 4: Apply JET colormap (blue=low, red=high activation)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    # Convert from BGR to RGB
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    
    return hm_colored


def create_superimposed_img(heatmap, pil_img, alpha=0.4):
    """
    Create superimposed image with improved heatmap overlay.
    
    Args:
        heatmap: Raw heatmap from Grad-CAM (0-1 normalized)
        pil_img: PIL Image of original specimen
        alpha: Transparency of heatmap overlay (0.4 default)
    
    Returns:
        dict with overlay and original images
    """
    orig = np.array(pil_img.convert("RGB"))
    h, w = orig.shape[:2]
    
    # Resize heatmap to original image size
    rsz = cv2.resize(heatmap, (w, h)) if heatmap.max() > 1e-8 \
          else np.zeros((h, w), dtype=np.float32)
    
    # Get improved colored heatmap using JET colormap
    hm_colored = create_improved_gradcam_heatmap(rsz)
    
    # Overlay with transparency
    overlay = cv2.addWeighted(
        orig.astype(np.float32), 1 - alpha,
        hm_colored.astype(np.float32), alpha,
        0
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return {
        "overlay": overlay,
        "heatmap_only": hm_colored,
        "side_by_side": np.concatenate([orig, overlay], axis=1)
    }
