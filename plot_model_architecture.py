from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

input_shape = (48, 48, 3)
# Create MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
# Add custom layers for facial expression detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

# Create the transfer learning model
model = Model(inputs=base_model.input, outputs=predictions)
# Plot the model architecture with labeled layers
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)