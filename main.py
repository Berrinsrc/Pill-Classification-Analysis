import tensorflow as tf
import keras
from keras import models,layers
import matplotlib.pyplot as plt 
import numpy as np
import os

data_path = 'Drug Vision/data combined'

train_ds= tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,   
    subset="training",      
    seed=123,               
    image_size=(128, 128),  
    batch_size=32
)

validation_ds= tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
        
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(), 
        
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'), 
        
        tf.keras.layers.Dense(10, activation='softmax') 
    ])
    return model

sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

optimizers_dict = {
    'Adam': 'adam',
    'RMSprop': 'rmsprop',
    'SGD + Momentum': sgd_optimizer
}

all_results = {}
epochs_to_run = 5

for opt_name, opt_object in optimizers_dict.items():
    print(f"\n{'='*50}")
    print(f">>> DENEY BAŞLIYOR: {opt_name} Optimizer")
    print(f"{'='*50}")
    
    model = create_model()
    model.compile(
        optimizer=opt_object,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds, 
        validation_data=validation_ds, 
        epochs=epochs_to_run
    )

    all_results[opt_name] = history.history['val_accuracy']

print("\n[BİLGİ] Eğitim tamamlandı. Grafik çizdiriliyor...")

plt.figure(figsize=(10, 6))

for opt_name in optimizers_dict.keys():
    plt.plot(range(1, epochs_to_run + 1), all_results[opt_name], label=opt_name, marker='o')

plt.title('Görüntü Sınıflandırma: Optimizer Karşılaştırması (Validation Accuracy)')
plt.xlabel('Eğitim Adımı (Epoch)')
plt.ylabel('Doğruluk Oranı (Validation Accuracy)')
plt.xticks(range(1, epochs_to_run + 1)) # X ekseninde sadece tam sayı epochları göster
plt.legend()
plt.grid(True)


plt.savefig('optimizer_karsilastirma_grafik.png')
print("\n>>> Başarılı! Grafik 'optimizer_karsilastirma_grafik.png' olarak kaydedildi.")
plt.show()