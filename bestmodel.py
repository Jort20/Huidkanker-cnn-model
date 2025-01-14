import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Metadata inladen
metadata = pd.read_csv("/students/2023-2024/Thema07/rumen_reactor/test/HAM10000_metadata.csv")
image_dir_part1 = "/students/2023-2024/Thema07/rumen_reactor/test/images/part_1"
image_dir_part2 = "/students/2023-2024/Thema07/rumen_reactor/test/images/part_2"

metadata['image_path'] = metadata['image_id'].apply(
    lambda x: os.path.join(
        image_dir_part1 if os.path.exists(os.path.join(image_dir_part1, f"{x}.jpg")) else image_dir_part2,
        f"{x}.jpg"
    )
)
metadata = metadata[metadata['image_path'].apply(os.path.exists)]

metadata.head()
metadata['age'].fillna((metadata['age'].mean()), inplace=True)
train, test = train_test_split(metadata, test_size=0.15, stratify=metadata['dx'], random_state=42)
train, val = train_test_split(train, test_size=0.15, stratify=train['dx'], random_state=42)

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
).flow_from_dataframe(
    train, x_col='image_path', y_col='dx', target_size=(224, 224),
    class_mode='categorical', batch_size=32
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    val, x_col='image_path', y_col='dx', target_size=(224, 224),
    class_mode='categorical', batch_size=32
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test, x_col='image_path', y_col='dx', target_size=(224, 224),
    class_mode='categorical', batch_size=32, shuffle=False
)

# Model initialiseren met BatchNormalization en L2 regularisatie
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(7, activation='softmax', kernel_regularizer=l2(0.01))(x)
model = Model(inputs=base_model.input, outputs=output)

# Mixed Precision Training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Compileer model
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)
# Model trainen
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50, 
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[early_stopping, reduce_lr],  # ModelCheckpoint verwijderd
    verbose=1
)
test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# Classificatierapport en Confusion Matrix
predictions = np.argmax(model.predict(test_gen), axis=-1)
true_labels = test_gen.classes

print(classification_report(true_labels, predictions, target_names=list(test_gen.class_indices.keys())))
conf_matrix = confusion_matrix(true_labels, predictions)
# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_gen.class_indices.keys(),
            yticklabels=test_gen.class_indices.keys())
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()


# Opslaan van het model voor toekomstig gebruik
model.save('/students/2023-2024/Thema07/rumen_reactor/test/skin_cancer_classifier.h5')
