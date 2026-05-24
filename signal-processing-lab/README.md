# Przetwarzanie Strumieni Danych (Signal Processing) - Lab

Projekt zawiera zestaw interaktywnych aplikacji (rozwiązań zadań z laboratoriów) m.in w Streamlit, realizujących zagadnienia z zakresu przetwarzania sygnałów cyfrowych.


## Struktura

L1 Signal Generation and basics
L2 Sampling and Reconstruction
L3 Signal Interpolaton Methonds
L4 Sampling and Interpolation of Signals
L5 Spectral Analysis and FFT
L6 Windowing and FFT
L7 Wavelet Transform and Signal Decomposition


## Tech

- Python 3.10+
- Streamlit
- NumPy
- SciPy
- Matplotlib
- Pandas
- PyWavelets (pywt)


## Implemented Concepts

- signal sampling and discrete-time representation
- aliasing effects and spectral distortion
- signal interpolation and reconstruction
- Fourier Transform (FFT) and frequency-domain analysis
- amplitude and phase spectra
- time-domain windowing techniques (window functions)
- spectral leakage and its mitigation
- windowed FFT analysis
- wavelet transform (DWT and CWT)
- multiresolution time–frequency analysis
- chirp signal analysis (non-stationary signals)
- spectral analysis of signals from external data (CSV input)

## RUN

```bash

pip install streamlit numpy scipy matplotlib pandas pywavelets

streamlit run LX/app.py