# Laboratorio 2 PDS 
## Introducción:

El procesamiento de señales utiliza herramientas clave como la convolución, que modela la interacción entre una señal y un sistema; la correlación, que mide la similitud entre señales; y la transformada, que permite analizar las señales en el dominio de la frecuencia. Estas técnicas, implementadas en Python, facilitan un análisis más profundo y práctico del tema

## Requerimientos:
- Interfaz de python (para este caso 3.12)
- Numpy
- Matplotlib
- Scipy.io
- Scipy.interpolate
- time
- IPython.display

##  Convolución:

La convolución es una operación que muestra cómo una señal interactúa con un sistema. En procesamiento de señales, se usa para calcular la salida de un sistema cuando se le aplica una entrada, ayudando a entender cómo el sistema afecta a la señal.

Para esta práctica, se implementaron dos métodos de convolución. El primero fue calculado y escrito a mano, como se muestra a continuación:

  ![Imagen de WhatsApp 2025-02-11 a las 19 24 51_da4792f8](https://github.com/user-attachments/assets/e19b8b4d-c752-4f82-bbfa-cdf2b0420f06)
Dentro de esta imagen se puede observar el calculo de la convolución de dos formas, como una multiplicación de polinomios y como una multiplicación en forma de tabla, ambas operaciones llegaron al mismo resultado, además en la parte final se encuentran las gráficas de la señal y del sistema respectivaente.







En python se representó de la siguiente manera con los valores solicitados en la guía: 

```bash
 
h = np.array([5,6,0,0,7,7,8]) #Señal de entrada [Código de estudiante]
x = np.array([1,0,7,6,2,4,2,2,3,7]) #Señal de salida [C.C]

p = np.array([5,6,0,0,7,8,6]) 
i = np.array([1,0,2,7,1,5,1,0,7,8])

#Se calcula la convolución.
y = np.convolve(h, x) 
yy=  np.convolve(p, i)

#Índices de señales.
npx = np.arange(len(p))
ni = np.arange(len(i))
nyy = np.arange(len(yy))

nh = np.arange(len(h))
nx = np.arange(len(x))
ny = np.arange(len(y))

```

Y de esta manera quedó graficado:

```bash
plt.figure(figsize=(10, 4),facecolor='linen')

plt.plot(nh, h, color='maroon')
plt.title('Señal h[n]  |Código Daniel|',color='darkslategray')
plt.xlabel('n' , color='darkslategray')
plt.ylabel('Amplitud', color='darkslategray')
plt.grid ()
plt.show() 

plt.figure(figsize=(10, 4),facecolor='linen')

plt.plot(nx, x, color='maroon')
plt.title('Señal x[n]  |C.C Daniel|',color='darkslategray')
plt.xlabel('n',color='darkslategray')
plt.ylabel('Amplitud',color='darkslategray')
plt.grid ()
plt.show()  

plt.figure(figsize=(10, 4),facecolor='linen')

plt.plot(ny, y, color='maroon')
plt.title('Señal y[n]=h[n] * x[n]  |Daniel|',color='darkslategray')
plt.xlabel('n',color='darkslategray')
plt.ylabel('Amplitud',color='darkslategray')
plt.grid ()
plt.show()

plt.figure(figsize=(10, 4),facecolor='linen')

plt.plot(npx, p, color='maroon')
plt.title('Señal p[n]  |Código Isabel|',color='darkslategray')
plt.xlabel('n' , color='darkslategray')
plt.ylabel('Amplitud', color='darkslategray')
plt.grid ()
plt.show() 

plt.figure(figsize=(10, 4),facecolor='linen')

plt.plot(ni, i, color='maroon')
plt.title('Señal i[n]  |C.C Isabel|',color='darkslategray')
plt.xlabel('n',color='darkslategray')
plt.ylabel('Amplitud',color='darkslategray')
plt.grid ()
plt.show()  

plt.figure(figsize=(10, 4),facecolor='linen')

plt.plot(nyy, yy, color='maroon')
plt.title('Señal yy[n]=h[n] * x[n]  |Isabel|',color='darkslategray')
plt.xlabel('n',color='darkslategray')
plt.ylabel('Amplitud',color='darkslategray')
plt.grid ()
plt.show()
```
# Gráficas de las convuluciones:

- Usando los datos de Daniel

![image](https://github.com/user-attachments/assets/fa5739ee-a46f-4c27-970c-6cf9b5ae0564)

![image](https://github.com/user-attachments/assets/624def76-bd5a-4be6-af28-40e18098d9fe)

![image](https://github.com/user-attachments/assets/ab6a1bf6-cf92-45be-a710-07b209f31f20)

- Usando los datos de Isabel

![image](https://github.com/user-attachments/assets/249231fd-f6ab-4b7e-8f10-0cb9e0a86e76)

![image](https://github.com/user-attachments/assets/5a3d5b70-ba0d-4213-8c31-b335a684d813)

![image](https://github.com/user-attachments/assets/b3fc2cbd-7c5b-45b2-8498-57e572f8a2d3)
## Señal electromiográfica de Physionet
Se eligió la señal EMG "emg_healthym" con duración de 10segundos en  PhysioNet, y se descargaron los archivos .info y .mat para su análisis. La electromiografía se tomó de un paciente masculino de 44 años sin antecedentes de enfermedad neuromuscular siendo así una EMG de electrodo de aguja concéntrico de 25 mm colocado en el músculo tibial anterior
El paciente dorsiflexionó el pie suavemente contra resistencia y luego lo relajó, todo esto se sabe gracias a la información que nos proporciona la pagina de Physionet.

- A continuación se muestran las librerias y el código implementado para la adecuación de la señal basado en los parametros descritos en el archivo .info descargado:

```bash
import matplotlib.pyplot as plt ## Crear gráficos y visualizaciones
import numpy as np  ## Manejo de arreglos y cálculos numéricos
from scipy.io import loadmat  ## Cargar archivos .mat 
from scipy.interpolate import make_interp_spline ## Interpolación y suavización de curvas
from scipy.signal import welch # Permite calcular y gráficar la densidad espectral

x=loadmat('emg_healthym.mat')  # ajuste de los valores segun el archivo .info
emg =np.transpose(emg) # transpone el vector de columnas a filas
emg = emg.squeeze() # Arregla las dimensiones en tamaño de la señal
fs =4000 # frecuencia de muestreo
tm =1/fs # tiempo entre muestras

````

- Así se calcularon los estadisticos descriptivos de la señal:

```bash
# Calcular la media aritmética manualmente
n = emg.size

if n > 1:
    suma = 0.0
    for x in emg:
        suma += x
    media = suma / n

# Calcular la desviación estándar manualmente
    suma1 = 0.0
    for x in emg:
        suma1 += (x - media) ** 2
    desvi = (suma1 / (n - 1)) ** 0.5  
else:
    media = float('error al calcular')
    desvi = float('error al calcular')

# Estadísticos calculados por medio de funciones
mediac = np.mean(emg)
desviacionc = np.std(emg, ddof=1) 

## coeficiente de variación calculado

coefi= desvi/media 

## coeficiente de variación con los valores de las funciones

coefi1= desviacionc/mediac

print(f"\nMedia calculada: {media}\n")
print(f"Desviación estándar calculada: {desvi}\n")
print(f"coeficiente de variación calculado: {coefi}\n")
print(f"Media por funciones: {mediac}\n")
print(f"Desviación estándar por funciones: {desviacionc}\n")
print(f"coeficiente de variación con valores de las funciones: {coefi1}\n")
```
- Así se graficó la señal EMG en función del tiempo:
  
```bash
#Graficamos la señal con respecto al tiempo:
tiempo=np.linspace(0,desviacionc,len(emg))
plt.figure(figsize=(10,4),facecolor='linen')
plt.plot(tiempo, emg, label="Señal en el tiempo",color='indianred')
plt.xlabel("Tiempo (s)", color='darkslategray')
plt.ylabel("Amplitud" , color='darkslategray')
plt.title("Señal en el Dominio del Tiempo", color='darkslategray')
plt.legend()
plt.grid()
plt.show()
```
- Gráfica señal Electromiográfica:

![image](https://github.com/user-attachments/assets/59ef6c41-4334-4a81-a468-ddda7aca18d7)



## Transformada de Fourier: 
La Transformada de Fourier descompone una señal en sus componentes de frecuencia, permitiendo su análisis en el dominio de la frecuencia y así se analizó con la señal EMG.

```bash
#Se cre la transformada de Fourier
N = len(emg)
frecuencias = np.fft.fftfreq(N, d=1/fs)
trs_magnitud = np.abs(np.fft.fft(emg))

#Se grafica la transformada de Fourier con respecto a la señal EMG
plt.figure(figsize=(10, 4),facecolor='linen')
plt.plot(frecuencias, trs_magnitud, label="Magnitud de la Transformada de fourier",color='indianred')
plt.xlabel("Frecuencia [Hz]", color='darkslategray')
plt.ylabel("Magnitud" , color='darkslategray')
plt.title("Transformada de Fourier de la Señal", color='darkslategray')
plt.grid()
plt.legend()
plt.show()
```

- Gráfica de la transformada de Fourier:

![image](https://github.com/user-attachments/assets/be47a9f8-5493-4b0b-af09-462cb59381f0)

## Densidad espectral:

La densidad espectral se puede definir cómo  se distribuye la potencia o energía de una señal en el dominio de la frecuencia, aplicando en Python se realizó de la siguiente manera.
```bash
frecuen_psd, psd= welch(emg, fs, nperseg=(4000))
plt.figure(figsize=(10, 4),facecolor='linen')
plt.semilogy(frecuen_psd, psd,label="Densidad Espectral",color='indianred' )
plt.xlabel("Frecuencia [Hz]", color='darkslategray')
plt.ylabel("Densidad de potencia" , color='darkslategray')
plt.title("Densidad Espectral de la señal", color='darkslategray')
plt.legend()
plt.show()
```
- Gráfica densidad espectral:

  ![image](https://github.com/user-attachments/assets/3baf40d8-9243-4bf7-bd9e-38ada01d31e3)


