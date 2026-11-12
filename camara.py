import cv2
import numpy as np
import tensorflow as tf

# 1) Load your Keras model
model = tf.keras.models.load_model("ssl_model.keras")

# 2) Inspect what input size it expects (height, width, channels)
print("Model input:", model.input_shape)   # e.g. (None, 224, 224, 3)
print("Model output:", model.output_shape) # e.g. (None, 10)

# Pick target size from model input
_, H, W, C = model.input_shape

cam = cv2.VideoCapture(0)  # on mac you might try cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ok, frame = cam.read()
    if not ok:
        print("Error reading frame")
        break

    # OpenCV gives BGR; many models are trained on RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model expected size
    img = cv2.resize(rgb, (W, H))

    # Convert to float32 and normalize (common: 0..1)
    x = img.astype(np.float32) / 255.0

    # Add batch dimension: (1, H, W, C)
    x = np.expand_dims(x, axis=0)

    # Predict
    y = model.predict(x, verbose=0)

    # Example: show top score
    score = float(np.max(y))
    cls = int(np.argmax(y))
    cv2.putText(frame, f"class={cls} score={score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()