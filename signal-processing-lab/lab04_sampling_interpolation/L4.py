import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURACJA STRON§Y STREAMLIT ---
st.set_page_config(page_title="Przetwarzanie Strumieni Danych - Lista 4", layout="wide")

st.title("Lista 4: Próbkowanie i Interpolacja Sygnałów")
st.markdown("---")

# --- PANEL BOCZNY (WIDŻETY) ---
st.sidebar.header("Parametry Sygnału")
# f = częstotliwość sygnału, fs = częstotliwość próbkowania
f = st.sidebar.slider("Częstotliwość sygnału (f) [Hz]", min_value=1.0, max_value=20.0, value=3.0, step=0.5)
fs = st.sidebar.slider("Częstotliwość próbkowania (fs) [Hz]", min_value=5.0, max_value=100.0, value=20.0, step=1.0)

# Czas trwania sygnału do analizy (1 sekunda)
T = 1.0 

# --- GENERACJA BAZOWYCH DANYCH ---
# 1. Sygnał ciągły (tzw. "analogowy") - bardzo wysoka rozdzielczość, udaje ideał
t_cont = np.linspace(0, T, 5000)
x_cont = np.sin(2 * np.pi * f * t_cont)

# 2. Sygnał spróbkowany (dyskretny) - punkty z częstotliwością fs
t_samp = np.arange(0, T, 1/fs)
x_samp = np.sin(2 * np.pi * f * t_samp)

# =========================================================================
# ZADANIE 1: Próbkowanie
# =========================================================================
st.header("Zadanie 1: Sygnał sinusoidalny i jego próbki")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(t_cont, x_cont, 'k-', alpha=0.5, label=f'Sygnał ciągły ($f={f}$ Hz)')
ax1.stem(t_samp, x_samp, linefmt='r-', markerfmt='ro', basefmt='k-', label=f'Próbki ($f_s={fs}$ Hz)')
ax1.set_xlabel("Czas [s]")
ax1.set_ylabel("Amplituda")
ax1.set_xlim([0, T])
ax1.legend(loc="upper right")
ax1.grid(True, linestyle='--', alpha=0.7)

st.pyplot(fig1)

# =========================================================================
# ZADANIE 2: Interpolacja Liniowa
# =========================================================================
st.markdown("---")
st.header("Zadanie 2: Interpolacja Liniowa (Odcinkowa)")

# np.interp wykonuje interpolację liniową (łączy punkty prostymi liniami)
x_lin = np.interp(t_cont, t_samp, x_samp)

# Błąd interpolacji: Sygnał oryginalny - Sygnał zrekonstruowany
error_lin = x_cont - x_lin

# Używamy kolumn Streamlita dla lepszego ułożenia (Wykres po lewej, Błąd po prawej)
col1, col2 = st.columns(2)

with col1:
    fig2a, ax2a = plt.subplots(figsize=(8, 4))
    ax2a.plot(t_cont, x_cont, 'k-', alpha=0.3, label='Oryginał', linewidth=3)
    ax2a.plot(t_cont, x_lin, 'b--', label='Interpolacja liniowa')
    ax2a.plot(t_samp, x_samp, 'ro', markersize=4)
    ax2a.set_title("Rekonstrukcja Liniowa")
    ax2a.grid(True, linestyle='--', alpha=0.6)
    ax2a.legend()
    st.pyplot(fig2a)

with col2:
    fig2b, ax2b = plt.subplots(figsize=(8, 4))
    ax2b.plot(t_cont, error_lin, 'r-', label='Błąd (Liniowa)')
    ax2b.fill_between(t_cont, error_lin, 0, color='red', alpha=0.2)
    ax2b.set_title("Błąd Interpolacji Liniowej")
    ax2b.grid(True, linestyle='--', alpha=0.6)
    ax2b.legend()
    st.pyplot(fig2b)

# =========================================================================
# ZADANIE 3: Interpolacja Whittakera-Shannona (Sinc)
# =========================================================================
st.markdown("---")
st.header("Zadanie 3: Interpolacja Whittakera-Shannona (Sinc)")

# Implementacja wzoru Whittakera-Shannona z definicji sumy:
# x(t) = sum( x[n] * sinc((t - nTs)/Ts) )
# W Pythonie np.sinc(x) jest zdefiniowane jako sin(pi*x)/(pi*x), więc nie mnożymy przez pi!

x_sinc = np.zeros_like(t_cont)
for n, x_n in enumerate(x_samp):
    # argument dla sinc: (t - n*Ts)/Ts = t/Ts - n = t*fs - n
    x_sinc += x_n * np.sinc(t_cont * fs - n)

# Błąd interpolacji Sinc
error_sinc = x_cont - x_sinc

col3, col4 = st.columns(2)

with col3:
    fig3a, ax3a = plt.subplots(figsize=(8, 4))
    ax3a.plot(t_cont, x_cont, 'k-', alpha=0.3, label='Oryginał', linewidth=3)
    ax3a.plot(t_cont, x_sinc, 'g--', label='Interpolacja Sinc')
    ax3a.plot(t_samp, x_samp, 'ro', markersize=4)
    ax3a.set_title("Rekonstrukcja Whittakera-Shannona")
    ax3a.grid(True, linestyle='--', alpha=0.6)
    ax3a.legend()
    st.pyplot(fig3a)

with col4:
    fig3b, ax3b = plt.subplots(figsize=(8, 4))
    ax3b.plot(t_cont, error_sinc, 'g-', label='Błąd (Sinc)')
    ax3b.fill_between(t_cont, error_sinc, 0, color='green', alpha=0.2)
    ax3b.set_title("Błąd Interpolacji Sinc")
    ax3b.grid(True, linestyle='--', alpha=0.6)
    ax3b.legend()
    st.pyplot(fig3b)

# --- PODSUMOWANIE (BŁĘDY W LICZBACH) ---
st.markdown("---")
st.subheader("Podsumowanie Błędów (Max)")
st.metric("Max błąd Liniowej", f"{np.max(np.abs(error_lin)):.4f}")
st.metric("Max błąd Sinc", f"{np.max(np.abs(error_sinc)):.4f}")