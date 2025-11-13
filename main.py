import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. Load MNIST (training)
# -----------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# -----------------------
# 2. Build simple CNN
# -----------------------
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=1, batch_size=128)

# -----------------------
# 3. Adversarial example (FGSM)
# -----------------------
loss_fn = keras.losses.SparseCategoricalCrossentropy()

def fgsm(model, x, y, eps=0.1):
    x_var = tf.cast(x, tf.float32)
    y_var = tf.cast([y], tf.int64)

    with tf.GradientTape() as tape:
        tape.watch(x_var)
        pred = model(x_var)
        loss = loss_fn(y_var, pred)

    grad = tape.gradient(loss, x_var)
    signed_grad = tf.sign(grad)

    x_adv = x_var + eps * signed_grad
    x_adv = tf.clip_by_value(x_adv, 0, 1)

    return x_adv

# Pick one MNIST test sample
i = 0
x_orig = x_test[i:i+1]
y_orig = y_test[i]

pred_orig = np.argmax(model.predict(x_orig))

# Create adversarial version
x_adv = fgsm(model, x_orig, y_orig, eps=0.15)
pred_adv = np.argmax(model.predict(x_adv))

print("Original label:", y_orig)
print("Pred (original):", pred_orig)
print("Pred (adversarial):", pred_adv)

# Visualize
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.title(f"Original (pred={pred_orig})")
plt.imshow(x_orig[0,:,:,0], cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Adversarial (pred={pred_adv})")
plt.imshow(x_adv[0,:,:,0], cmap="gray")
plt.axis("off")
plt.show()

# -----------------------
# 4. OOD example (Fashion-MNIST)
# -----------------------
(_, _), (x_fmnist, y_fmnist) = keras.datasets.fashion_mnist.load_data()
x_fmnist = x_fmnist.astype("float32") / 255.0
x_fmnist = np.expand_dims(x_fmnist, -1)

# Pick a random fashion item (e.g., a shirt)
x_ood = x_fmnist[0:1]
y_ood_true = y_fmnist[0]

pred_ood = np.argmax(model.predict(x_ood))

print("\n OOD INPUT (Fashion-MNIST):")
print("\n True clothing label:", y_ood_true)
print("\n MNIST model predicted digit:", pred_ood)

plt.figure()
plt.title(f"OOD example (model predicts digit={pred_ood})")
plt.imshow(x_ood[0,:,:,0], cmap="gray")
plt.axis("off")
plt.show()
