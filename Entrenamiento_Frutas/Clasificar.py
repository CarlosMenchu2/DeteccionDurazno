import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout,Flatten,Dense,Activation
from tensorflow.python.keras.layers import Convolution2D,MaxPooling2D
from tensorflow.python.keras import backend as k
import matplotlib.pyplot as plt

#*********************************** Directorio de las imagenes de entrenamiento***************************************
imagenes_entrenamiento = 'C:/Users/carlo/Desktop/Entrenamiento_Frutas/pruebas1/imgduraznos' #debe ser modificado dependiendo de donde se encuentren las imagenes
#**********************************************************************************************************************


epocas = 20
altura = 100
longitud = 100
batch_size = 32
pasos = 1500
canales = 3
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool = (2,2)
clases = 5
lr = 0.0005

entrenamiento_imagenes = ImageDataGenerator(
    rescale=1./255,
)

imagen_entrenamiento = entrenamiento_imagenes.flow_from_directory(
    
    imagenes_entrenamiento,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'

)

print("Indices: ")
print(imagen_entrenamiento.class_indices)

red_neuronal = Sequential()
red_neuronal.add(Convolution2D(filtrosConv1,tamano_filtro1,padding='same',input_shape=(altura,longitud,canales),activation='relu'))
red_neuronal.add(MaxPooling2D(pool_size=tamano_pool))
red_neuronal.add(Convolution2D(filtrosConv2,tamano_filtro2,padding='same',activation='relu'))
red_neuronal.add(MaxPooling2D(pool_size=tamano_pool))

red_neuronal.add(Flatten())
#cnn.add(Dense(256,activation='relu'))
#cnn.add(Dense(315,activation='relu'))
#cnn.add(Dense(516,activation='relu'))
#cnn.add(Dense(555,activation='relu')) #Modelo 7
red_neuronal.add(Dense(650,activation='relu'))#Modelo 10,11
#red_neuronal.add(Dense(525,activation='relu'))
red_neuronal.add(Dropout(0.5))
red_neuronal.add(Dense(clases,activation='softmax'))
red_neuronal.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])

historial = red_neuronal.fit(imagen_entrenamiento,steps_per_epoch=pasos,epochs=epocas)
print(imagen_entrenamiento.class_indices)

plt.xlabel("# Epoca")
plt.ylabel("ECM")
plt.plot(historial.history['loss'])


direccion = 'C:/Users/carlo/Desktop/Entrenamiento_Frutas/Modelo/'

if not os.path.exists(direccion):
    os.mkdir(direccion)

red_neuronal.save('C:/Users/carlo/Desktop/Entrenamiento_Frutas/Modelo/modelo_P1.h5')
red_neuronal.save_weights('C:/Users/carlo/Desktop/Entrenamiento_Frutas/Modelo/pesos_P1.h5')