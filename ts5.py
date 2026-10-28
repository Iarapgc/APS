import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt   
import scipy.io as sio
import pandas as pd

plt.close('all')

#%% ECG

##################
# Lectura de ECG #
##################
fs_ecg = 1000 # Hz

##################
## ECG sin ruido
##################

ecg=np.load('ecg_sin_ruido.npy')

# plt.figure()
# plt.plot(ecg)
# plt.title("ECG sin ruido")

#nperseg:largo de cada segmento.
frec_8,pxx_8=sig.welch(ecg,fs_ecg,nperseg=len(ecg)//8)#Da segmentos de tamaño entero
frec_10,pxx_10=sig.welch(ecg,fs_ecg,nperseg=len(ecg)//10)#Da segmentos de tamaño entero
frec_15,pxx_15=sig.welch(ecg,fs_ecg,nperseg=len(ecg)//15)#Da segmentos de tamaño entero
pxx_db_8=10*np.log10(pxx_8)
pxx_db_10=10*np.log10(pxx_10)
pxx_db_15=10*np.log10(pxx_15)

#%%% Ancho de Banda
#Devuelve la suma de valores a lo largo de un vector
pot_acum_nor_8=np.cumsum(pxx_8)/np.sum(pxx_8) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_8=frec_8[np.argmax(pot_acum_nor_8>=0.99)]
#print(ancho_8)

pot_acum_nor_15=np.cumsum(pxx_15)/np.sum(pxx_15) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_15=frec_15[np.argmax(pot_acum_nor_15>=0.99)]
#print(ancho_15)

pot_acum_nor_10=np.cumsum(pxx_10)/np.sum(pxx_10) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_10=frec_10[np.argmax(pot_acum_nor_10>=0.99)]
#print(ancho_10)

# Límite superior del zoom [Hz]
f_zoom = 1.2 * ancho_10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax in (ax1, ax2):
    ax.plot(frec_10, pxx_db_10, color='blue', linewidth=1, alpha=0.6, label="10 segmentos")
    ax.plot(frec_8, pxx_db_8, color='green', linewidth=1, alpha=0.6, label="8 segmentos")
    ax.plot(frec_15, pxx_db_15, color='hotpink', linewidth=1, alpha=0.6, label="15 segmentos")
    ax.axvline(ancho_10, color='k', linestyle='--', linewidth=1,
               label=f'Ancho de banda 99 % = {ancho_10:.1f} Hz')
    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('PSD [dB/Hz]')
    ax.grid(True, linestyle=':', alpha=0.6)

# Gráfico 1: espectro completo
ax1.set_title('PSD por método de Welch - ECG')
ax1.set_xlim(0, fs_ecg / 2)

# Gráfico 2: zoom en los primeros picos
ax2.set_title(f'Zoom: 0 a {f_zoom:.0f} Hz')
ax2.set_xlim(0, f_zoom)
mask = frec_10 <= f_zoom   # reajusta el eje y al rango visible
ax2.set_ylim(pxx_db_10[mask].min() - 5, pxx_db_10[mask].max() + 5)

# Para poner una única leyenda debajo de los gráficos porque sino tapa información
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
fig.tight_layout(rect=[0, 0.08, 1, 1])   # Para que la leyenda no quede pegada a los ejes
plt.show()

ancho_ecg_selec=ancho_10

#%% PPG

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')
# plt.figure()
# plt.plot(ppg)

#nperseg:largo de cada segmento.
frec_8,pxx_8=sig.welch(ppg,fs_ppg,nperseg=len(ppg)//8)#Da segmentos de tamaño entero
frec_10,pxx_10=sig.welch(ppg,fs_ppg,nperseg=len(ppg)//10)#Da segmentos de tamaño entero
frec_5,pxx_5=sig.welch(ppg,fs_ppg,nperseg=len(ppg)//5)#Da segmentos de tamaño entero
pxx_db_8=10*np.log10(pxx_8)
pxx_db_10=10*np.log10(pxx_10)
pxx_db_5=10*np.log10(pxx_5)

#%%% Ancho de Banda
#Devuelve la suma de valores a lo largo de un vector
pot_acum_nor_8=np.cumsum(pxx_8)/np.sum(pxx_8) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_8=frec_8[np.argmax(pot_acum_nor_8>=0.99)]
#print(ancho_8)

pot_acum_nor_5=np.cumsum(pxx_5)/np.sum(pxx_5) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_5=frec_5[np.argmax(pot_acum_nor_5>=0.99)]
#print(ancho_5)

pot_acum_nor_10=np.cumsum(pxx_10)/np.sum(pxx_10) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_10=frec_10[np.argmax(pot_acum_nor_10>=0.99)]
#print(ancho_10)

# Límite superior del zoom [Hz]: ajustalo mirando el primer gráfico
f_zoom = 1.2 * ancho_8
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax in (ax1, ax2):
    ax.plot(frec_10, pxx_db_10, color='blue', linewidth=1, alpha=0.6, label="10 segmentos")
    ax.plot(frec_8, pxx_db_8, color='green', linewidth=1, alpha=0.6, label="8 segmentos")
    ax.plot(frec_5, pxx_db_5, color='hotpink', linewidth=1, alpha=0.6, label="5 segmentos")
    ax.axvline(ancho_8, color='k', linestyle='--', linewidth=1,
               label=f'Ancho de banda 99 % = {ancho_8:.1f} Hz')
    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('PSD [dB/Hz]')
    ax.grid(True, linestyle=':', alpha=0.6)

# Gráfico 1: espectro completo
ax1.set_title('PSD por método de Welch - PPG')
ax1.set_xlim(0, fs_ppg / 2)

# Gráfico 2: zoom en los primeros picos
ax2.set_title(f'Zoom: 0 a {f_zoom:.0f} Hz')
ax2.set_xlim(0, f_zoom)
mask = frec_8 <= f_zoom   # reajusta el eje y al rango visible
ax2.set_ylim(pxx_db_8[mask].min() - 5, pxx_db_8[mask].max() + 5)

# Para poner una única leyenda debajo de los gráficos porque sino tapa información
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
fig.tight_layout(rect=[0, 0.08, 1, 1])   # Para que la leyenda no quede pegada a los ejes
plt.show()

ancho_ppg_selec=ancho_8

#%% Sonido

####################
# Lectura de audio #
####################

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# plt.figure()
# plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)
# ================= Normalizo la potencia a 1W =================
wav_data=wav_data/np.sqrt(np.var(wav_data))
print("Wav data - Potencia del audio normalizado")
print(np.var(wav_data))
# ==============================================================

#%%% Diagrama espectral


#El número indica la cantidad de segmentos en los que se dividió el intervalo
#Por defecto usa la ventana Hann para ventanear los segmentos

frec_30,pxx_30=sig.welch(wav_data,fs_audio,nperseg=len(wav_data)//30)#Da segmentos de tamaño entero
#nperseg:largo de cada segmento.
frec_50,pxx_50=sig.welch(wav_data,fs_audio,nperseg=len(wav_data)//50)#Da segmentos de tamaño entero
frec_40,pxx_40=sig.welch(wav_data,fs_audio,nperseg=len(wav_data)//40)#Da segmentos de tamaño entero
#sig.welch devuelve potencia, así que para logaritmarlo no hay que elevar al cuadrado

pxx_db_30=10*np.log10(pxx_30)
pxx_db_50=10*np.log10(pxx_50)
pxx_db_40=10*np.log10(pxx_40)

#%%% Ancho de banda
"""
Usando este método del porcentaje de potencia es necesario aclarar que si hay mucho ruido en las frecuencias bajas y la información real está en las frecuencias altas, con este método parte de la información se va a perder
"""

#Devuelve la suma de valores a lo largo de un vector
pot_acum_nor_30=np.cumsum(pxx_30)/np.sum(pxx_30) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_30=frec_30[np.argmax(pot_acum_nor_30>=0.99)] #acá se puede justificar con el rango audible y cómo recortaría yo más en un estilo pasabanda
#print(ancho_30)

pot_acum_nor_40=np.cumsum(pxx_40)/np.sum(pxx_40) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_40=frec_40[np.argmax(pot_acum_nor_40>=0.99)] #acá se puede justificar con el rango audible y cómo recortaría yo más en un estilo pasabanda
#print(ancho_40)

pot_acum_nor_50=np.cumsum(pxx_50)/np.sum(pxx_50) #Suma acumulativa en un vector/suma total. Lo normaliza
ancho_50=frec_50[np.argmax(pot_acum_nor_50>=0.99)] #acá se puede justificar con el rango audible y cómo recortaría yo más en un estilo pasabanda
#print(ancho_50)

# Límite superior del zoom [Hz]: ajustalo mirando el primer gráfico
f_zoom = 1.2 * ancho_40
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax in (ax1, ax2):
    ax.plot(frec_50, pxx_db_50, color='blue', linewidth=1, alpha=0.6, label="50 segmentos")
    ax.plot(frec_30, pxx_db_30, color='green', linewidth=1, alpha=0.6, label="30 segmentos")
    ax.plot(frec_40, pxx_db_40, color='hotpink', linewidth=1, alpha=0.6, label="40 segmentos")
    ax.axvline(ancho_40, color='k', linestyle='--', linewidth=1,
               label=f'Ancho de banda 99 % = {ancho_40:.1f} Hz')
    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('PSD [dB/Hz]')
    ax.grid(True, linestyle=':', alpha=0.6)

# Gráfico 1: espectro completo
ax1.set_title('PSD por método de Welch - la cucaracha.wav')
ax1.set_xlim(0, fs_audio / 2)

# Gráfico 2: zoom en los primeros picos
ax2.set_title(f'Zoom: 0 a {f_zoom:.0f} Hz')
ax2.set_xlim(0, f_zoom)
mask = frec_40 <= f_zoom   # reajusta el eje y al rango visible
ax2.set_ylim(pxx_db_40[mask].min() - 5, pxx_db_40[mask].max() + 5)

# Para poner una única leyenda debajo de los gráficos porque sino tapa información
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
fig.tight_layout(rect=[0, 0.08, 1, 1])   # Para que la leyenda no quede pegada a los ejes
plt.show()
ancho_sonido_selec=ancho_40

#%% Tabla comparativa de ancho de banda

tabla = pd.DataFrame({
    'Señal':        ['ECG', 'PPG', 'Audio (la cucaracha.wav)'],
    'fs [Hz]':      [fs_ecg, fs_ppg, fs_audio],
    'BW 99 % [Hz]': [ancho_ecg_selec, ancho_ppg_selec, ancho_sonido_selec],
})
tabla['BW / (fs/2) [%]'] = 100 * tabla['BW 99 % [Hz]'] / (tabla['fs [Hz]'] / 2)
print("\nTabla Comparativa de BW:\n")
print(tabla.round(1).to_string(index=False))