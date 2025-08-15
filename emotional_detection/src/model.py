import os
import tensorflow as tf
from preprocessing import create_data_generators
from model_architecture import CNN_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_DIR = os.path.dirname(BASE_DIR)                

TRAIN_DATA_DIR  = os.path.join(PROJECT_DIR, "data", "train")
TEST_DATA_DIR   = os.path.join(PROJECT_DIR, "data", "test")
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "models", "model_VGG.h5")


class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
epochs = 50
batch_size = 32

def count_images(directory):
    return sum(len(files) for _, _, files in os.walk(directory) if files)

def train_model():
    train_gen, test_gen = create_data_generators(TRAIN_DATA_DIR, TEST_DATA_DIR)

    model = CNN_model(input_shape=(48, 48, 1),
                      num_classes=len(train_gen.class_indices))
    
    num_train = count_images(TRAIN_DATA_DIR)
    num_test = count_images(TEST_DATA_DIR)

    print("==" * 60)
    print(f"Training images: {num_train}")
    print("--" * 60)
    print(f"Testing images: {num_test}")
    print("==" * 60)

    model.fit(
        train_gen,
        steps_per_epoch=num_train // batch_size,
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=num_test // batch_size,
        batch_size=batch_size,
        verbose=1
    )

    model.save(MODEL_SAVE_PATH)
    print(f" Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
