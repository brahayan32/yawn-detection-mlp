import numpy as np
import tensorflow as tf


class PureTensorFlowMLP(tf.Module):
    """MLP built with TensorFlow variables, without Keras."""

    def __init__(self, input_dim: int = 6400, seed: int = 42, name: str | None = None):
        super().__init__(name=name)
        tf.random.set_seed(seed)

        self.w1 = tf.Variable(tf.random.normal([input_dim, 256], stddev=0.05), name="w1")
        self.b1 = tf.Variable(tf.zeros([256]), name="b1")
        self.w2 = tf.Variable(tf.random.normal([256, 64], stddev=0.05), name="w2")
        self.b2 = tf.Variable(tf.zeros([64]), name="b2")
        self.w3 = tf.Variable(tf.random.normal([64, 1], stddev=0.05), name="w3")
        self.b3 = tf.Variable(tf.zeros([1]), name="b3")

    @property
    def trainable_variables(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def __call__(self, x):
        hidden_1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.w2) + self.b2)
        return tf.sigmoid(tf.matmul(hidden_2, self.w3) + self.b3)


class AdamOptimizer:
    """Minimal Adam optimizer implemented with TensorFlow operations."""

    def __init__(self, variables, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = tf.Variable(0.0, trainable=False)
        self.m = [tf.Variable(tf.zeros_like(var), trainable=False) for var in variables]
        self.v = [tf.Variable(tf.zeros_like(var), trainable=False) for var in variables]

    def apply_gradients(self, gradients, variables):
        self.t.assign_add(1.0)
        for i, (gradient, variable) in enumerate(zip(gradients, variables)):
            if gradient is None:
                continue
            self.m[i].assign(self.beta1 * self.m[i] + (1.0 - self.beta1) * gradient)
            self.v[i].assign(self.beta2 * self.v[i] + (1.0 - self.beta2) * tf.square(gradient))
            m_hat = self.m[i] / (1.0 - tf.pow(self.beta1, self.t))
            v_hat = self.v[i] / (1.0 - tf.pow(self.beta2, self.t))
            variable.assign_sub(self.learning_rate * m_hat / (tf.sqrt(v_hat) + self.epsilon))


def binary_cross_entropy(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(loss)


def l2_penalty(model):
    weights = [variable for variable in model.trainable_variables if variable.shape.rank and variable.shape.rank > 1]
    if not weights:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n([tf.reduce_sum(tf.square(weight)) for weight in weights])


def accuracy(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
    y_label = tf.cast(y_pred >= 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_label), tf.float32))


def train_step(model, optimizer, x_batch, y_batch, l2_strength: float = 0.0):
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        loss = binary_cross_entropy(y_batch, y_pred)
        if l2_strength:
            loss = loss + l2_strength * l2_penalty(model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(gradients, model.trainable_variables)
    return loss, accuracy(y_batch, y_pred)


def evaluate_model(model, x, y):
    y_pred = model(tf.convert_to_tensor(x, dtype=tf.float32))
    return {
        "loss": float(binary_cross_entropy(y, y_pred).numpy()),
        "accuracy": float(accuracy(y, y_pred).numpy()),
    }


def predict(model, x):
    return model(tf.convert_to_tensor(x, dtype=tf.float32)).numpy().reshape(-1)


def train_mlp(
    model,
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 6,
    l2_strength: float = 0.0,
):
    optimizer = AdamOptimizer(model.trainable_variables, learning_rate=learning_rate)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    has_validation = x_val is not None and len(x_val)
    best_weights = [var.numpy().copy() for var in model.trainable_variables]
    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(len(x_train))
        train_losses, train_accuracies = [], []

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = tf.convert_to_tensor(x_train[batch_idx], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_train[batch_idx], dtype=tf.float32)
            loss_value, acc_value = train_step(model, optimizer, x_batch, y_batch, l2_strength=l2_strength)
            train_losses.append(float(loss_value.numpy()))
            train_accuracies.append(float(acc_value.numpy()))

        history["loss"].append(float(np.mean(train_losses)))
        history["accuracy"].append(float(np.mean(train_accuracies)))

        if has_validation:
            val_metrics = evaluate_model(model, x_val, y_val)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            print(
                f"Epoca {epoch}/{epochs} - loss: {history['loss'][-1]:.4f} - "
                f"accuracy: {history['accuracy'][-1]:.4f} - val_loss: {val_metrics['loss']:.4f} - "
                f"val_accuracy: {val_metrics['accuracy']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_weights = [var.numpy().copy() for var in model.trainable_variables]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Entrenamiento detenido por Early Stopping.")
                    break
        else:
            print(f"Epoca {epoch}/{epochs} - loss: {history['loss'][-1]:.4f} - accuracy: {history['accuracy'][-1]:.4f}")

    if has_validation:
        for variable, value in zip(model.trainable_variables, best_weights):
            variable.assign(value)

    return history


def save_model(model, output_dir):
    tf.saved_model.save(model, str(output_dir))


def load_model(model_dir):
    return tf.saved_model.load(str(model_dir))
