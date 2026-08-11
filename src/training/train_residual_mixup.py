import tensorflow as tf, numpy as np, os, random, glob, argparse

from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from src.training.custom_callback import ICBHI_Score_PrintingCallback
from src.models.custom_cnn_residual import create_custom_cnn


# ============================================================
# MIXUP MODEL WRAPPER
# ============================================================
class MixupModel(tf.keras.Model):
    def __init__(self, *args, mixup_alpha=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixup_alpha = mixup_alpha

    def train_step(self, data):
        X, y = data

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # Shuffle indices for mixing
        batch_size = tf.shape(X)[0]
        indices = tf.random.shuffle(tf.range(batch_size))

        # Mix inputs and soft labels
        X_mix = lam * X + (1.0 - lam) * tf.gather(X, indices)
        y_mix = lam * y + (1.0 - lam) * tf.gather(y, indices)

        with tf.GradientTape() as tape:
            y_pred = self(X_mix, training=True)
            loss = self.compiled_loss(y_mix, y_pred)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Métricas con etiquetas reales (no mezcladas)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


class ICBHIEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, min_delta=1e-6):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get('icbhi_score', -np.inf)

        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True
                print(f"\nEarly stopping: best ICBHI Score = {self.best_score:.5f}")


def get_label(file_path):
    """
    Devuelve etiqueta multietiqueta [crackle, wheeze] a partir del nombre de carpeta.

    Healthy          → [0, 0]
    Crackle          → [1, 0]
    Wheeze           → [0, 1]
    Wheeze & Crackle → [1, 1]
    """
    elements = tf.strings.split(file_path, os.path.sep)
    label_str = elements[5]

    # Etiqueta Crackle
    is_crackle = tf.cast(
        tf.logical_or(
            tf.equal(label_str, 'Crackle'),
            tf.equal(label_str, 'Wheeze & Crackle')
        ), tf.float32
    )

    # Etiqueta Wheeze
    is_wheeze = tf.cast(
        tf.logical_or(
            tf.equal(label_str, 'Wheeze'),
            tf.equal(label_str, 'Wheeze & Crackle')
        ), tf.float32
    )

    return tf.stack([is_crackle, is_wheeze])


def spec_augment_tf(spec, time_mask_param=20, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
    H = tf.shape(spec)[0]
    W = tf.shape(spec)[1]

    for _ in range(num_freq_masks):
        f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, H - f + 1, dtype=tf.int32)
        mask = tf.ones_like(spec)
        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=tf.reshape(tf.range(f0, f0 + f), (-1, 1)),
            updates=tf.zeros((f, W, tf.shape(spec)[2]))
        )
        spec = spec * mask

    for _ in range(num_time_masks):
        t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, W - t + 1, dtype=tf.int32)
        time_mask = tf.concat([
            tf.ones((H, t0, tf.shape(spec)[2])),
            tf.zeros((H, t, tf.shape(spec)[2])),
            tf.ones((H, W - t0 - t, tf.shape(spec)[2]))
        ], axis=1)
        spec = spec * time_mask

    return spec


def get_class_weights_from_paths(dir_dataset):
    """
    Calcula pesos por etiqueta (crackle, wheeze) de forma independiente.
    Devuelve un array [w_crackle, w_wheeze] con pesos balanceados.
    """
    train_paths = sorted(glob.glob(os.path.join(dir_dataset, 'Train/*/*')))

    crackle_labels = []
    wheeze_labels = []

    for path in train_paths:
        folder = path.split(os.path.sep)[-2]
        crackle_labels.append(1 if folder in ('Crackle', 'Wheeze & Crackle') else 0)
        wheeze_labels.append(1 if folder in ('Wheeze', 'Wheeze & Crackle') else 0)

    crackle_labels = np.array(crackle_labels)
    wheeze_labels  = np.array(wheeze_labels)

    w_crackle = compute_class_weight('balanced', classes=np.array([0, 1]), y=crackle_labels)
    w_wheeze  = compute_class_weight('balanced', classes=np.array([0, 1]), y=wheeze_labels)

    return np.array([w_crackle[1], w_wheeze[1]])  # peso de la clase positiva


def load_npy(path):
    path = path.numpy().decode("utf-8")
    spec = np.load(path)
    assert spec.shape[1] == 113, f"Unexpected spectrogram width: {spec.shape[1]}"
    return spec.astype(np.float32)


def process_npy(file_path, training=True):
    # Etiqueta multietiqueta [crackle, wheeze]
    label = get_label(file_path)
    label = tf.ensure_shape(label, [2])

    spec = tf.py_function(load_npy, [file_path], tf.float32)
    spec.set_shape([128, 113, 1])

    if training:
        spec = tf.cond(
            tf.random.uniform([]) < 0.8,
            lambda: spec_augment_tf(spec, time_mask_param=20, freq_mask_param=15, num_time_masks=2, num_freq_masks=2),
            lambda: spec
        )

    return spec, label


def load_datasets(dir_dataset, seed=12345):
    train_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Train/*/*'), shuffle=False)
    test_dataset  = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Test/*/*'),  shuffle=False)

    options = tf.data.Options()
    options.experimental_deterministic = True
    train_dataset = train_dataset.with_options(options)
    test_dataset  = test_dataset.with_options(options)

    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), seed=seed, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(lambda x: process_npy(x, training=True),  num_parallel_calls=1)
    train_dataset = train_dataset.batch(128, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1)

    test_dataset = test_dataset.map(lambda x: process_npy(x, training=False), num_parallel_calls=1)
    test_dataset = test_dataset.batch(128, drop_remainder=True)
    test_dataset = test_dataset.prefetch(1)

    return train_dataset, test_dataset


def train(model, train_dataset, val_dataset):
    icbhi_callback = ICBHI_Score_PrintingCallback(val_dataset)

    # ============================================================
    # PHASE 1 — INITIAL TRAINING
    # ============================================================
    print("PHASE 1: INITIAL TRAINING (lr=1e-3)")
    early_stopping_1 = ICBHIEarlyStopping(patience=5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    model.fit(train_dataset, epochs=15, validation_data=val_dataset, verbose=2,
              callbacks=[icbhi_callback, early_stopping_1])

    # ============================================================
    # PHASE 2 — PARTIAL FINE-TUNING
    # ============================================================
    print("PHASE 2: REFINE TRAINING (lr=1e-4)")
    early_stopping_2 = ICBHIEarlyStopping(patience=15)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    model.fit(train_dataset, epochs=50, validation_data=val_dataset, verbose=2,
              callbacks=[icbhi_callback, early_stopping_2])

    # ============================================================
    # PHASE 3 — FULL FINE-TUNING
    # ============================================================
    print("PHASE 3: FINE-TUNING (lr=1e-5)")
    early_stopping_3 = ICBHIEarlyStopping(patience=20)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=2,
              callbacks=[icbhi_callback, early_stopping_3])

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================
    model.save(os.path.join('models', datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.keras'))

    return model


# ============================================================
# ENTRY POINT
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=12345)
args = parser.parse_args()

SEED = args.random_seed

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

print("Loading datasets...")
train_dataset, val_dataset = load_datasets('/app/data/processed', seed=SEED)

print("Creating model...")
# num_classes=2: salida [crackle, wheeze] con sigmoid
base_model = create_custom_cnn(input_shape=(128, 113, 1), num_classes=2, seed=SEED)

# Wrap with MixupModel to override train_step
model = MixupModel(inputs=base_model.input, outputs=base_model.output, mixup_alpha=0.3)

print("Starting training...")
model = train(model, train_dataset, val_dataset)