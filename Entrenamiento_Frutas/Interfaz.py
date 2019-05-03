from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image
from PIL import ImageTk
from resizeimage import resizeimage
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array
import cv2

nombre_imagen=""
logitud,altura = 100,100
#************************ Carga  del modelo y los pesos para la deteccion del estado de madurez del durazno*******************************
modelo='C:/Users/carlo/Desktop/Entrenamiento_Frutas/Modelo/modelo_10.h5'#esta ruta deben modificarse por la direccion donde este el modelo_10.h5
pesos= 'C:/Users/carlo/Desktop/Entrenamiento_Frutas/Modelo/pesos_10.h5' #esta ruta debe modificarse por la direccion donde esten los pesos_10.h5
cnn = tf.keras.models.load_model(modelo)
cnn.load_weights(pesos)
#**********************************************************************************************************************************************

#****************************************************Modelo que Verificar si la imagen es un durazno*******************************************************
def ver_durazno(imagen):

    logitud,altura = 100,100
    #*********************Carga del modelo de deteccion si la imagen es de un durazno o no************************************************************************
    modelo='C:/Users/carlo/Desktop/Entrenamiento_Frutas/Modelo/modelo_Deteccion.h5'#esta ruta deben modificarse por la direccion donde este el modelo_Deteccion.h5
    pesos= 'C:/Users/carlo/Desktop/Entrenamiento_Frutas/Modelo/pesos_Deteccion.h5' #esta ruta debe modificarse por la direccion donde esten los pesos_Deteccion.h5
    cnn = tf.keras.models.load_model(modelo)
    cnn.load_weights(pesos)
    #*************************************************************************************************************************************************************
    def predict(file):
        x = load_img(file,target_size=(logitud,altura))
        x = img_to_array(x)
        x=np.expand_dims(x,axis=0)
        arreglo = cnn.predict(x) # se envia la imagen al modelo para la prediccion
        print("Prediccion: ",arreglo)
        resultado = arreglo[0]#obtenemos la prediccion del modelo
        respuesta = np.argmax(resultado)#obtememos el porcentaje mas alto que retorno el modelo

        if respuesta == 0:
            return True
        elif respuesta==1:
            return True
        elif respuesta ==2:
            return False

    if (predict(imagen)==True): #Si la imagen es un durazno se envia al metodo mostrar resultado que es el encargado de determinar el estado de 
        mostrar_resultado(imagen)#maduracion en que se encuentra

    else: #si no es un durazno despliega un mensaje indicando que no lo es

        estadoDurazno.delete('1.0', END)
        estadoDurazno.insert(INSERT, "No es un DURAZNO")

#**********************************************************************************************************************************************

#*********************************************** Determinacion de la calidad del durazno*******************************************************
def calidadDurazno(file):

    #azulBajo = np.array([115,100,20],np.uint8)
    #azulAlto = np.array([135,255,255],np.uint8)

    azulBajo = np.array([108,100,100],np.uint8)
    azulAlto = np.array([128,255,255],np.uint8)


    rosadoBajo = np.array([157,100,100],np.uint8)
    rosadoAlto = np.array([177,255,255],np.uint8)

    cont_azul=0
    cont_rosa=0
    img = cv2.imread(file)
    imagen = cv2.resize(img,(500,550))
    #img = resizeimage.resize_thumbnail(img,[400,400])
    cv2.imshow('frame',imagen)
    figuraHSV = cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)#Convertimos la imagen de BGR A HSV
    #------------------------------Deteccion de Color Azul-----------------------------------
    maskAzul = cv2.inRange(figuraHSV,azulBajo,azulAlto)
    _,contornos,_ = cv2.findContours(maskAzul,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #----------------------------------------------------------------------------------------

    #-----------------------------Deteccion de color Rosado----------------------------------
    maskRosado = cv2.inRange(figuraHSV,rosadoBajo,rosadoAlto)
    _,contornosRosados,_ = cv2.findContours(maskRosado,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #----------------------------------------------------------------------------------------

    #-----------------------------Identificacion de los contornos de color azul--------------

    for c in contornos:
        area = cv2.contourArea(c)
        if area > 10:
            cv2.drawContours(imagen,[c],0,(255,0,0),3)
            cont_azul =cont_azul+1
    #-----------------------------------------------------------------------------------------
    #-----------------------------Identificacion de los contornos de color Rosado-------------
    for v in contornosRosados:
        area1 = cv2.contourArea(v)
        if area1 >10:
            cv2.drawContours(imagen,[v],0,(13,197,23),3)
            cont_rosa = cont_rosa+1
    
    calidad = ((cont_azul/2)) +((cont_azul/2)-1) 
    #print("Cont azul: ",cont_azul," Cont Rosa: ",cont_verde)
    print("Cont azul: ",cont_azul)
    print("Calidad: ",calidad)

    if calidad < 5:
        return "Calidad Del Durazno: Corriente"+'\n'+"Diametro(cm): "+str(calidad)
    elif (calidad==5) or (calidad<7):
        return "Calidad del Durazno: Segunda"+'\n'+"Diametro(cm): "+str(calidad)
    elif calidad >7:
        return "Calidad del Durazno: Primera"+'\n'+"Diametro(cm): "+str(calidad)
    #----------------------------------------------------------------------------------------
#**********************************************************************************************************************************************

#***********************************************Modelo que verifica estado de maduracion del durazno**************************************
def mostrar_resultado(file):
    estadoMaduracion = "Tipo de Fruta: Durazno"+'\n'    
    x = load_img(file,target_size=(logitud,altura))
    x = img_to_array(x)
    x=np.expand_dims(x,axis=0)
    arreglo = cnn.predict(x)# se envia la imagen al modelo para la prediccion
    print("Prediccion: ",arreglo)
    resultado = arreglo[0]#obtenemos la prediccion del modelo
    respuesta = np.argmax(resultado)#obtememos el porcentaje mas alto que retorno el modelo

    if respuesta == 0:
        print("Durazno Maduro")
        estadoMaduracion += "Estado de Maduracion: Maduro "+'\n'
        estadoMaduracion += "Tiene un tiempo de 2 a 4 dias para su consumo"+'\n'
        estadoMaduracion += calidadDurazno(file)
        estadoDurazno.delete('1.0', END)
        estadoDurazno.insert(INSERT, estadoMaduracion)
    elif respuesta==1:
        print("Durazno Podrido Fase 1")
        estadoMaduracion += "Estado de Maduracion: Podrido Fase 1"+'\n'
        estadoMaduracion += "El durazno ya no es comestible"+'\n'
        estadoMaduracion += calidadDurazno(file)
        estadoDurazno.delete('1.0', END)
        estadoDurazno.insert(INSERT, estadoMaduracion)
    elif respuesta==2:
        print("Durazno Podrido Fase 2")
        estadoMaduracion += "Estado de Maduracion: Podrido Fase 2"+'\n'
        estadoMaduracion += "El durazno ya no es comestible"+'\n'
        estadoDurazno.delete('1.0', END)
        estadoDurazno.insert(INSERT, estadoMaduracion)
    elif respuesta==3:
        print("Durazno Verde")
        estadoMaduracion += "Estado de Maduracion: Verde"+'\n'
        estadoMaduracion += "Tiene un tiempo de 4-8 dias para su consumo"+'\n'
        estadoMaduracion += calidadDurazno(file)
        estadoDurazno.delete('1.0', END)
        estadoDurazno.insert(INSERT, estadoMaduracion)
    elif respuesta==4:
        print("Durazno Sobre Maduro")
        estadoMaduracion += "Estado de Maduracion: Sobre Maduro"+'\n'
        estadoMaduracion += "Tiene un tiempo de 2 a 3 dias para su consumo"+'\n'
        estadoMaduracion += calidadDurazno(file)
        estadoDurazno.delete('1.0', END)
        estadoDurazno.insert(INSERT, estadoMaduracion)
#**********************************************************************************************************************************************


#******************************************************* CREACION DE INTERFAZ GRAFICA**********************************************************
raiz = Tk()
raiz.title("Determinacion de Maduracion del Durazno")
formulario = Frame(raiz,width=700,height=600,bg="black")
formulario.pack()

#***************** TXT donde se muentra la ruta de la imagen seleccionada **************
txt = Entry(formulario,width=60,font=("Helvetica","10"))
txt.place(x=160,y=60)
#***************************************************************************************

titulo = Label(formulario,text="Determinacion del Estado de Madurez de un Durazno",font=("Helvetica","14"),bg="black",fg="white")
titulo.place(x=0,y=0)
titulo.config(padx=125,pady=5)


estado = Label(formulario,text="Resultado del Analisis:",font=("Arial","14"),bg="black",fg="white")
estado.place(x=50,y=370)
#estado.config(padx=20,pady=20)

#************* Creacion del contendor de informacion del estado del Durazno*************
estadoDurazno = Text(formulario,width=62,height=10,font=("Helvetica","12"))
estadoDurazno.place(x=50,y=400)
#***************************************************************************************

#******************Imagen Inicial del Programa******************************************
img = Image.open("url.png")
img = resizeimage.resize_thumbnail(img,[200,150])
photoimg = ImageTk.PhotoImage(img)
imagenDurazno = Label(formulario,image=photoimg,bg="black")
imagenDurazno.place(x=240,y=150)
#***************************************************************************************

#**************** Abre un cuadro de dialogo para buscar la imagen a analizar************
def buscar_imagen():
    txt.delete(0,END)
    nombre_imagen = askopenfilename()
    print(nombre_imagen)
    txt.insert(0,nombre_imagen)
    return nombre_imagen
#***************************************************************************************

#************** Actualiza la Imagen inicial a la escogida por el Usuario****************
def abrir_imagen():
    x = buscar_imagen()
    img1 = Image.open(x)
    img1 = resizeimage.resize_thumbnail(img1,[200,200])
    photoimg1 = ImageTk.PhotoImage(img1)
    imagenDurazno.config(image=photoimg1)
    imagenDurazno.image = photoimg1
#***************************************************************************************

#************************** Buacar la Imagen que sera analizada por la red**************
boton_Imagen = Button(formulario,text = "Buscar Imagen",command=abrir_imagen,bg="dark slate blue",fg="white",font=("Helvetica","12"))
boton_Imagen.place(x=35,y=55)
#***************************************************************************************

#***************** Boton que llama a la red neuronal que analiza el durazno*************
#primero pasa por el modelo que detecta si es un Durazno
#Si lo es envia la imagen al modelo que determina el estado de maduracion en el que se encuentra
#Tambien se detecta la calidad del Durazno
boton_analizar = Button(formulario,text = "Analizar",font=(12),bg="red",fg="white",padx=35,command=lambda:ver_durazno(txt.get()))
boton_analizar.place(x=471,y=355)
#***************************************************************************************

raiz.mainloop()
#*******************************************************************************************************************************************************

