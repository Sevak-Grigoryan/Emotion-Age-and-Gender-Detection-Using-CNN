import os
import tensorflow as tf
from preprocesing import create_data_generators
from model_architecture import CNN_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DATA_DIR  = os.path.join(BASE_DIR, "data", "Training")
TEST_DATA_DIR   = os.path.join(BASE_DIR, "data", "Validation")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "model.keras")

class_labels = ["Female", "Male"] 
epochs = 30
batch_size = 32

def count_images(directory):
    return sum(len(files) for _, _, files in os.walk(directory) if files)

def train_model():
    # չենք փոխում Ձեր preprocessing-ի ստորագրությունը/կառուցվածքը
    train_gen, test_gen = create_data_generators(
        TRAIN_DATA_DIR, TEST_DATA_DIR,
        target_size=(48, 48),
        batch_size=batch_size
    )

    # եթե Ձեր preprocessing-ը վերադարձնում է categorical (one-hot) նաև 2 դասի դեպքում,
    # սա կդարձնի ելքային չափը 2 (softmax)։
    num_classes = len(train_gen.class_indices)
    print("Class indices:", train_gen.class_indices)  # օրինակ՝ {'female': 0, 'male': 1}

    # մոդելի input_shape-ը grayscale-ի համար (48,48,1) է
    model = CNN_model(input_shape=(48, 48, 1), num_classes=num_classes)

    # loss/metrics — categorical 2-դասի համար OK է
    # (եթե ձեր CNN_model-ը փոխված է sigmoid-ով 1 ելքով, օգտագործեք binary_crossentropy)
    loss_fn = "categorical_crossentropy" if num_classes > 1 else "binary_crossentropy"
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=loss_fn,
                  metrics=["accuracy"])

    num_train = count_images(TRAIN_DATA_DIR)
    num_test  = count_images(TEST_DATA_DIR)

    print("==" * 60)
    print(f"Training images: {num_train}")
    print("--" * 60)
    print(f"Validation images: {num_test}")
    print("==" * 60)

    model.fit(
        train_gen,
        steps_per_epoch=max(1, num_train // batch_size),
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=max(1, num_test // batch_size)
    )

    model.save(MODEL_SAVE_PATH)
    print(f" Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()

