# DeteccionDurazno
Proyecto del curso de inteligencia artificial 2019 - Determinación del estado de madurez del Durazno

Se deben de descargar los siguientes archivos: 

Archivos del modelo de determinación de la madurez del durazno:

•	"modelo_10.h5"
•	"pesos_10.h5"

Archivos del modelo de detección del durazno:

•	"modelo_Deteccion.h5"
•	"pesos_Deteccion.h5"


estos archivos no fueron incluidos en el repositorio debido a que son demasiado pesados y github no perimitio subirlos porque
indicaba que superaban el limite maximo permitido, por lo cual fueron subidos a Google Drive,
el link de descarga es:  https://drive.google.com/open?id=1SnnNOiNWP93lYWiFM5P-06ee1Spy7B2b

despues de descargar estos archivos deben de pegarse en la carpeta "Modelo" y la ruta absoluta de estos se debe de modificar
en el archivo "Interfaz.py" de la siguiente manera:

•	En el  método "Ver_durazno()" debe ir la ruta absoluta de "modelo_Deteccion.h5" y "pesos_Deteccion.h5"
•	También debe de ir la ruta absoluta de "modelo_10.h5" y "pesos_10.h5" en la parte principal de programa

En el codigo esta especificado de mejor manera donde se debe de modificar.

