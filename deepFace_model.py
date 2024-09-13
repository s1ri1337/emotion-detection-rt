from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

def resnet_model_creation(input_shape=(48, 48, 3), num_classes=7):
    # Load the ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Fine-tune the entire ResNet model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model