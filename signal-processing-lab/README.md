# Przetwarzanie Strumieni Danych (Signal Processing) - Lab

Projekt zawiera zestaw interaktywnych aplikacji i skryptów realizujących zaawansowane zagadnienia z zakresu cyfrowego przetwarzania sygnałów (DSP). Rozwiązania zaimplementowano z użyciem bibliotek analitycznych języka Python oraz interaktywnych dashboardów w frameworku Streamlit.

## Struktura Projektu

| Katalog | Opis | Interfejs |
|---|---|---|
| `lab01_signal_generation_and_visualization` | Generacja sygnałów, szumów, histogramy, zapis/odczyt CSV | Matplotlib |
| `lab02_image_processing` | Przetwarzanie obrazów, histogramy RGB, szumy (Rician, Poisson), rotacje | Matplotlib / OpenCV |
| `lab03_psd_analysis` | Widmowa Gęstość Mocy (PSD), Periodogram, metoda Welcha, z definicji | Matplotlib |
| `lab04_sampling_interpolation` | Próbkowanie, aliasing, interpolacja liniowa i Whittakera-Shannona (Sinc) | Streamlit |
| `lab05_frequency_spectrum_analysis` | Analiza widmowa (FFT), widma amplitudowe i fazowe, obwiednia | Streamlit |
| `lab06_windowing_and_fft` | Funkcje okna (Dirichlet, Hann, Blackman), przeciek widma (spectral leakage) | Streamlit |
| `lab07_wavelet_transform_and_decomposition` | Transformata falkowa (DWT/CWT), dekompozycja i rekonstrukcja | Streamlit |
| `lab08_spectrograms_and_EMD` | Spektrogramy (STFT) oraz Empiryczna Dekompozycja na Mody (EMD) | Streamlit |
| `lab09_denoise_and_quality_metric` | Miary jakości (SNR, PSNR, MSE) oraz filtry (Wiener, Savitzky-Golay, EMD) | Streamlit |

## Technologie (Tech Stack)

* Python 3.10+
* Streamlit
* NumPy
* SciPy
* Pandas
* Matplotlib
* OpenCV (cv2)
* PyWavelets (pywt)
* EMD (Empirical Mode Decomposition)
* Scikit-Learn

## Zrealizowane Koncepcje (Implemented Concepts)

* Generacja, próbkowanie i dyskretna reprezentacja sygnałów
* Efekt aliasingu i zniekształcenia widmowe
* Rekonstrukcja sygnałów (interpolacja Sinc i liniowa)
* Transformata Fouriera (FFT), analiza w dziedzinie częstotliwości, widma amplitudowe i fazowe
* Obliczanie Widmowej Gęstości Mocy (PSD)
* Techniki okienkowania w czasie i minimalizacja przecieku widma
* Przetwarzanie obrazów 2D (histogramy barw, szum Sól i Pieprz, transformacje geometryczne)
* Analiza sygnałów niestacjonarnych (chirp, spektrogramy)
* Transformata falkowa i wielorozdzielcza analiza czasowo-częstotliwościowa
* Empiryczna Dekompozycja na Mody (algorytm EMD i funkcje IMF)
* Odszumianie sygnałów z wykorzystaniem zaawansowanych filtrów cyfrowych
* Estymacja miar jakości sygnału (Stosunek sygnału do szumu, błąd średniokwadratowy)
* Analiza widmowa sygnałów z zewnętrznych zbiorów danych (wejście CSV)

## Uruchomienie (RUN)

Zainstaluj wymagane zależności:

```bash
pip install streamlit numpy scipy matplotlib pandas opencv-python pywavelets emd scikit-learn
```

### Uruchamiania Laboratoriów 1-2 w interfejsach Matplotlib (Przykład lab01)

```bash
python L1.py
```

### Uruchamiania Laboratoriów 4-9 - aplikacji Streamlit (Przykład lab09)

```bash
streamlit run L9.py
```