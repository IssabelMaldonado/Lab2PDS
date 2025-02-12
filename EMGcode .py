import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import make_interp_spline
from scipy.signal import welch
from IPython.display import display, clear_output 
import time

# Definimos las señales originales
h = np.array([5,6,0,0,7,7,8])  # Señal de entrada (Código Daniel)
x = np.array([1,0,7,6,2,4,2,2,3,7])  # Señal de salida (C.C)

p = np.array([5,6,0,0,7,8,6])  
i = np.array([1,0,2,7,1,5,1,0,7,8])

# Se calculan las convoluciones
y = np.convolve(h, x)  
yy = np.convolve(p, i)

# Índices para las señales
nh = np.arange(len(h))
nx = np.arange(len(x))
ny = np.arange(len(y))

npx = np.arange(len(p))
ni = np.arange(len(i))
nyy = np.arange(len(yy))

# Función para graficar señales estáticas
def graficar_senal(n, s, titulo):
    plt.figure(figsize=(10, 4), facecolor='linen')
    plt.plot(n, s, color='maroon')
    plt.title(titulo, color='darkslategray')
    plt.xlabel('n', color='darkslategray')
    plt.ylabel('Amplitud', color='darkslategray')
    plt.grid()
    plt.show()

# Graficar las señales originales y convoluciones
graficar_senal(nh, h, 'Señal h[n]  |Código Daniel|')
graficar_senal(nx, x, 'Señal x[n]  |C.C Daniel|')
graficar_senal(ny, y, 'Señal y[n]=h[n] * x[n]  |Daniel|')
graficar_senal(npx, p, 'Señal p[n]  |Código Isabel|')
graficar_senal(ni, i, 'Señal i[n]  |C.C Isabel|')
graficar_senal(nyy, yy, 'Señal yy[n]=p[n] * i[n]  |Isabel|')

# -------------------- FUNCIÓN PARA ANIMAR CONVOLUCIÓN --------------------

def animar_convolucion(x_signal, c_signal, y_signal, titulo):
    len_x = len(x_signal)
    len_c = len(c_signal)
    len_y = len(y_signal)

    plt.figure(figsize=(10, 6))

    for i in range(len_y):
        plt.clf()

        # Graficar x[n]
        plt.subplot(3, 1, 1)
        plt.stem(np.arange(len_x), x_signal, linefmt='b-', markerfmt='bo', basefmt="black")
        plt.title(f"{titulo} - Señal de Entrada x[n]")
        plt.xlabel("n")
        plt.grid()

        # Graficar c[n] desplazado correctamente
        plt.subplot(3, 1, 2)
        inicio = max(0, i - len_c + 1)  # Controla el desplazamiento del sistema
        desplazamiento = np.zeros(inicio)
        c_desplazado = np.concatenate([desplazamiento, c_signal])[:i+1]
        plt.stem(np.arange(i+1), c_desplazado, linefmt='r-', markerfmt='ro', basefmt="black")
        plt.title(f"{titulo} - Sistema c[n] desplazado (Paso {i+1})")
        plt.xlabel("n")
        plt.grid()

        # Graficar y[n] acumulado correctamente
        plt.subplot(3, 1, 3)
        plt.stem(np.arange(i + 1), y_signal[:i + 1], linefmt='g-', markerfmt='go', basefmt="black")
        plt.title(f"{titulo} - Salida y[n] acumulada")
        plt.xlabel("n")
        plt.grid()

        plt.tight_layout()
        
        # Mostrar la animación correctamente
        clear_output(wait=True)  # Borra la gráfica anterior
        display(plt.gcf())  # Muestra la nueva gráfica
        time.sleep(0.3)  # Pausa breve para animación fluida

    # Limpiar la última iteración
    clear_output(wait=True)

# -------------------- ANIMACIÓN PARA DANIEL --------------------
animar_convolucion(h, x, y, "Convolución Daniel")

# -------------------- ANIMACIÓN PARA ISABEL --------------------
animar_convolucion(p, i, yy, "Convolución Isabel")

### correlación 

# Definimos parámetros
Fs = 1 / (1.25e-3)  # Frecuencia de muestreo = 1 / Ts
Ts = 1 / Fs         # Periodo de muestreo
f = 100             # Frecuencia de la señal
n = np.arange(0, 9) # Valores de n

# Definimos las señales
x1 = np.cos(2 * np.pi * f * n * Ts)
x2 = np.sin(2 * np.pi * f * n * Ts)

# Calculamos la correlación cruzada
correlacion = np.correlate(x1, x2, mode="full")
lags = np.arange(-len(x1) + 1, len(x1))

# ----- GRAFICAMOS LAS SEÑALES -----
plt.figure(figsize=(12, 6))

# Gráfico de x1[n]
plt.subplot(3, 1, 1)
plt.stem(n, x1, linefmt='b-', markerfmt='bo', basefmt="black", label="x1[n] = cos(2π100nTs)")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.title("Señal x1[n]")
plt.grid()
plt.legend()

# Gráfico de x2[n]
plt.subplot(3, 1, 2)
plt.stem(n, x2, linefmt='r-', markerfmt='ro', basefmt="black", label="x2[n] = sin(2π100nTs)")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.title("Señal x2[n]")
plt.grid()
plt.legend()

# Gráfico de la correlación
plt.subplot(3, 1, 3)
plt.stem(lags, correlacion, linefmt='g-', markerfmt='go', basefmt="black", label="Correlación x1[n] y x2[n]")
plt.xlabel("Desplazamiento (lags)")
plt.ylabel("Amplitud")
plt.title("Correlación Cruzada entre x1[n] y x2[n]")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# ----- IMPRIMIMOS LOS RESULTADOS -----
print("Valores de la correlación cruzada:")
print(correlacion)



# Cargamos los datos de PhysioNet
x = loadmat('emg_healthym.mat')

# Normalización de la señal EMG
emg = (x['val'] - 0) / 10000
emg = np.transpose(emg)
fs = 4000  # Frecuencia de muestreo
tm = 1 / fs  # Periodo de muestreo

# Estadísticos descriptivos
dr = len(emg) / fs
minn = np.min(emg)
maxx = np.max(emg)
med = np.mean(emg)
medi = np.median(emg)
vari = np.var(emg)
dev = np.std(emg)

print("Estadísticos descriptivos:")
print(f"Duración [s]: {dr}")
print(f"Mínimo: {minn}")
print(f"Máximo: {maxx}")
print(f"Media: {med}")
print(f"Mediana: {medi}")
print(f"Varianza: {vari}")
print(f"Desviación estándar: {dev}")

# Graficamos la señal EMG en el dominio del tiempo
tiempo = np.linspace(0, dr, len(emg))

plt.figure(figsize=(10, 4), facecolor='linen')
plt.plot(tiempo, emg, label="Señal en el tiempo", color='indianred')
plt.xlabel("Tiempo (s)", color='darkslategray')
plt.ylabel("Amplitud", color='darkslategray')
plt.title("Señal EMG en el Dominio del Tiempo", color='darkslategray')
plt.legend()
plt.grid()
plt.show()

# Transformada de Fourier
N = len(emg)
frecuencias = np.fft.fftfreq(N, d=1/fs)
trs_magnitud = np.abs(np.fft.fft(emg))

# Graficamos la Transformada de Fourier
plt.figure(figsize=(10, 4), facecolor='linen')
plt.plot(frecuencias[:N//2], trs_magnitud[:N//2], label="Magnitud de la Transformada de Fourier", color='indianred')
plt.xlabel("Frecuencia [Hz]", color='darkslategray')
plt.ylabel("Magnitud", color='darkslategray')
plt.title("Transformada de Fourier de la Señal EMG", color='darkslategray')
plt.grid()
plt.legend()
plt.show()

# Densidad espectral de potencia (PSD)
frecuen_psd, psd = welch(emg.flatten(), fs, nperseg=256)

plt.figure(figsize=(10, 4), facecolor='linen')
plt.semilogy(frecuen_psd, psd, label="Densidad Espectral de Potencia (PSD)", color='indianred')
plt.xlabel("Frecuencia [Hz]", color='darkslategray')
plt.ylabel("Densidad de Potencia", color='darkslategray')
plt.title("Densidad Espectral de Potencia de la Señal EMG", color='darkslategray')
plt.grid()
plt.legend()
plt.show()

# -------------------- HISTOGRAMA DE AMPLITUDES --------------------
plt.figure(figsize=(10, 4), facecolor='beige')
plt.hist(emg.flatten(), bins=50, color='coral', alpha=0.7, edgecolor='maroon')
plt.xlabel("Amplitud", color='red')
plt.ylabel("Frecuencia", color='red')
plt.title("Histograma de Amplitudes de la Señal EMG", color='red')
plt.grid()
plt.show()

especpot= trs_magnitud **2
ptotal= np.sum(especpot)

medfreq= np.sum(frecuencias * especpot) / ptotal
medianafreq= frecuencias[np.searchsorted(np.cumsum(especpot), ptotal/2)]               
desvifreq= np.sqrt(np.sum((frecuencias - medfreq)**2*especpot)/ptotal)              

print("Estadísticos con Frecuencia:")
print(f"Media: {medfreq}")
print(f"Mediana: {medianafreq}")
print(f"Varianza: {desvifreq}")
