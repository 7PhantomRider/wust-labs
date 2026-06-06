import numpy as np
import pandas as pd


fs = 500
t = np.linspace(0, 2, fs * 2, endpoint=False)

wolny_sinus = np.sin(2 * np.pi * 2 * t)
szybki_sinus = 0.5 * np.sin(2 * np.pi * 30 * t)
szum = 0.2 * np.random.randn(len(t))

#superpozycja
sygnal = wolny_sinus + szybki_sinus + szum


df = pd.DataFrame({
    'Czas': t,
    'Sygnal': sygnal
})

df.to_csv('testowy_sygnal.csv', index=False)
print("Plik 'testowy_sygnal.csv' został wygenerowany")
