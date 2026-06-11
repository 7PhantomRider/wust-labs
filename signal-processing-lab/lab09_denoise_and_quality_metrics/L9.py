import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import mean_squared_error
import emd
import warnings

warnings.filterwarnings('ignore')

# ─── Konfiguracja aplikacji ───────────────────────────────────────────────────
st.set_page_config(page_title="Lista 9 - Odszumianie", layout="wide")
plt.style.use('dark_background') # Prosty, czysty ciemny motyw wykresów

st.title("Lista 9: Odszumianie Sygnałów i Miary Jakości")

# ─── ZAD 1: MIARY JAKOŚCI ─────────────────────────────────────────────────────
def oblicz_mse(s, y):
    # Mean Squared Error (Błąd Średniokwadratowy)
    return np.mean((s - y)**2)

def oblicz_psnr(s, y):
    # Peak Signal-to-Noise Ratio (Szczytowy stosunek sygnału do szumu)
    mse = oblicz_mse(s, y)
    if mse == 0:
        return float('inf')
    s_max = np.max(np.abs(s))
    return 20 * np.log10(s_max / np.sqrt(mse))

def oblicz_snr(s, szum):
    # Signal-to-Noise Ratio (Stosunek sygnału do szumu)
    rms_s = np.sqrt(np.mean(s**2))
    rms_n = np.sqrt(np.mean(szum**2))
    if rms_n == 0:
        return float('inf')
    return 20 * np.log10(rms_s / rms_n)

# ─── ZAD 3: GENERACJA SZUMU Z ZADANYM SNR ─────────────────────────────────────
def generuj_szum(s, docelowy_snr, typ_szumu="biały"):
    rms_s = np.sqrt(np.mean(s**2))
    docelowy_rms_n = rms_s / (10 ** (docelowy_snr / 20))
    
    if typ_szumu == "biały":
        szum = np.random.randn(len(s))
    elif typ_szumu == "browna":
        szum = np.cumsum(np.random.randn(len(s)))
        szum = szum - np.mean(szum) # usunięcie trendu
    else:
        szum = np.zeros(len(s))
        
    aktualny_rms_n = np.sqrt(np.mean(szum**2))
    if aktualny_rms_n > 0:
        szum = szum * (docelowy_rms_n / aktualny_rms_n)
        
    return szum

# ─── ZAD 4, 5, 6: FILTRY ──────────────────────────────────────────────────────
def filtr_wienera(y, rozmiar_okna):
    return signal.wiener(y, mysize=rozmiar_okna)

def filtr_savgol(y, rozmiar_okna, rzad_wielomianu):
    if rozmiar_okna % 2 == 0: rozmiar_okna += 1 
    if rozmiar_okna <= rzad_wielomianu: rozmiar_okna = rzad_wielomianu + 2
    if rozmiar_okna % 2 == 0: rozmiar_okna += 1
    return signal.savgol_filter(y, rozmiar_okna, rzad_wielomianu)

def filtr_emd(y, odrzucone_imf):
    imfs = emd.sift.sift(y)
    if imfs.ndim == 1:
        imfs = imfs.reshape(-1, 1)
        
    if odrzucone_imf >= imfs.shape[1]:
        odrzucone_imf = imfs.shape[1] - 1
        
    odszumiony = np.sum(imfs[:, odrzucone_imf:], axis=1)
    return odszumiony

def rysuj_wykres(ax, t, s_czysty, y_zaszumiony, y_odszumiony, tytul):
    ax.plot(t, y_zaszumiony, color="gray", alpha=0.5, label="Zaszumiony")
    ax.plot(t, s_czysty, color="white", lw=1.5, ls="--", alpha=0.5, label="Oryginał")
    ax.plot(t, y_odszumiony, color="cyan", lw=1.5, label="Odszumiony")
    
    mse = oblicz_mse(s_czysty, y_odszumiony)
    psnr = oblicz_psnr(s_czysty, y_odszumiony)
    
    ax.set_title(f"{tytul} | MSE: {mse:.4f} | PSNR: {psnr:.2f} dB")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

# ─── INTERFEJS ZAKŁADEK ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Zadania 1 i 2: Miary jakości",
    "Zadania 3-6: Odszumianie (Sygnały)",
    "Zadanie 7: Odszumianie z pliku CSV"
])

# ─── ZAKŁADKA 1 ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Weryfikacja miar jakości")
    st.info("Porównanie wzorów z definicji (Zad 1) z bibliotekami Pythona (Zad 2).")
    
    col1, col2, col3 = st.columns(3)
    typ_sygnalu = col1.selectbox("Sygnał testowy", ["Sinus", "Prostokąt", "Piła", "Świergotliwy"])
    czestotliwosc = col2.slider("Częstotliwość [Hz]", 1.0, 50.0, 5.0, step=1.0)
    poziom_szumu = col3.slider("Siła szumu (SNR) [dB]", -10.0, 40.0, 10.0, step=1.0)
    
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    
    if typ_sygnalu == "Sinus": s = np.sin(2 * np.pi * czestotliwosc * t)
    elif typ_sygnalu == "Prostokąt": s = signal.square(2 * np.pi * czestotliwosc * t)
    elif typ_sygnalu == "Piła": s = signal.sawtooth(2 * np.pi * czestotliwosc * t)
    else: s = signal.chirp(t, f0=czestotliwosc, f1=czestotliwosc*10, t1=1.0, method='linear')
        
    szum = generuj_szum(s, poziom_szumu, "biały")
    y = s + szum
    
    fig1, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(t, y, color="red", alpha=0.5, label="Sygnał + Szum")
    ax1.plot(t, s, color="cyan", lw=2, label="Czysty Sygnał")
    ax1.set_title(f"Sygnał testowy: {typ_sygnalu}")
    ax1.grid(True, alpha=0.3); ax1.legend()
    st.pyplot(fig1)
    
    c_lewa, c_prawa = st.columns(2)
    with c_lewa:
        st.write("**Nasza implementacja (Zadanie 1):**")
        st.write(f"MSE: {oblicz_mse(s, y):.6f}")
        st.write(f"PSNR: {oblicz_psnr(s, y):.6f} dB")
        st.write(f"SNR: {oblicz_snr(s, szum):.6f} dB")
        
    with c_prawa:
        st.write("**Funkcje biblioteczne (Zadanie 2):**")
        lib_mse = mean_squared_error(s, y)
        st.write(f"MSE (sklearn): {lib_mse:.6f}")
        lib_psnr = 20 * np.log10(np.max(np.abs(s)) / np.sqrt(lib_mse))
        st.write(f"PSNR (numpy): {lib_psnr:.6f} dB")
        lib_snr = 10 * np.log10(np.var(s) / np.var(szum))
        st.write(f"SNR (numpy): {lib_snr:.6f} dB")

# ─── ZAKŁADKA 2 ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Odszumianie wygenerowanych sygnałów")
    
    c1, c2, c3, c4 = st.columns(4)
    typ_syg = c1.selectbox("Sygnał", ["Świergotliwy", "Sinus", "Prostokąt"])
    typ_szumu = c2.selectbox("Szum", ["Biały", "Browna"])
    docelowy_snr = c3.slider("SNR [dB]", -10.0, 30.0, 5.0, step=1.0)
    fs2 = c4.number_input("Częstotliwość próbkowania (fs)", value=1000)
    
    t2 = np.linspace(0, 1, fs2, endpoint=False)
    if typ_syg == "Świergotliwy": s2 = signal.chirp(t2, f0=5, f1=100, t1=1.0, method='linear')
    elif typ_syg == "Sinus": s2 = np.sin(2 * np.pi * 10 * t2)
    else: s2 = signal.square(2 * np.pi * 5 * t2)
        
    szum2 = generuj_szum(s2, docelowy_snr, typ_szumu.lower())
    y2 = s2 + szum2
    
    st.write("---")
    fc1, fc2, fc3 = st.columns(3)
    okno_wiener = fc1.slider("Wiener: Rozmiar okna", 3, 99, 25, step=2)
    okno_sg = fc2.slider("Savitzky-Golay: Rozmiar okna", 5, 99, 31, step=2)
    rzad_sg = fc2.slider("Savitzky-Golay: Rząd wielomianu", 1, 5, 3)
    odrzuc_emd = fc3.slider("EMD: Odrzucone szybkie składowe (IMF)", 1, 5, 1)
    
    y_wiener = filtr_wienera(y2, okno_wiener)
    y_savgol = filtr_savgol(y2, okno_sg, rzad_sg)
    y_emd = filtr_emd(y2, odrzuc_emd)
    
    fig2, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 10))
    fig2.subplots_adjust(hspace=0.5)
    
    ax0.plot(t2, s2, color="white", lw=2, ls="--", alpha=0.5, label="Oryginał")
    ax0.plot(t2, y2, color="red", lw=1.0, alpha=0.8, label="Zaszumiony")
    ax0.set_title(f"Sygnał wejściowy | Zmierzone SNR: {oblicz_snr(s2, szum2):.2f} dB")
    ax0.grid(True, alpha=0.3); ax0.legend(loc="upper right")
    
    rysuj_wykres(ax1, t2, s2, y2, y_wiener, "Zadanie 4: Filtr Wienera")
    rysuj_wykres(ax2, t2, s2, y2, y_savgol, "Zadanie 5: Filtr Savitzky-Golay")
    rysuj_wykres(ax3, t2, s2, y2, y_emd, f"Zadanie 6: EMD (usunięto {odrzuc_emd} składowe)")
    
    st.pyplot(fig2)

# ─── ZAKŁADKA 3 ───────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Odszumianie sygnału z pliku CSV (Zadanie 7)")
    
    plik_csv = st.file_uploader("Wybierz plik .csv", type="csv")
    
    if plik_csv is not None:
        try:
            c1, c2 = st.columns(2)
            separator = c1.text_input("Separator kolumn", value=",")
            fs_csv = c2.number_input("Częstotliwość próbkowania (fs)", min_value=1, value=500)
            
            df = pd.read_csv(plik_csv, sep=separator)
            kolumny_numeryczne = df.select_dtypes(include=np.number).columns.tolist()
            
            if not kolumny_numeryczne:
                st.error("Brak kolumn liczbowych w pliku.")
            else:
                kol_y = st.selectbox("Kolumna z sygnałem zaszumionym:", kolumny_numeryczne)
                y_csv = np.ascontiguousarray(df[kol_y].dropna().values, dtype=np.float64).flatten()
                t_csv = np.arange(len(y_csv)) / fs_csv
                
                kol_s = st.selectbox("Kolumna z czystym sygnałem (opcjonalnie do błędów):", ["Brak"] + kolumny_numeryczne)
                if kol_s != "Brak":
                    s_csv = np.ascontiguousarray(df[kol_s].dropna().values, dtype=np.float64).flatten()
                else:
                    s_csv = y_csv 
                    
                if st.button("Uruchom odszumianie CSV"):
                    with st.spinner("Przetwarzanie danych..."):
                        y_w_csv = filtr_wienera(y_csv, 25)
                        y_sg_csv = filtr_savgol(y_csv, 31, 3)
                        
                        y_dither = y_csv + np.random.randn(len(y_csv)) * 1e-6
                        y_e_csv = filtr_emd(y_dither, 1)
                        
                        fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
                        fig3.subplots_adjust(hspace=0.5)
                        
                        rysuj_wykres(ax1, t_csv, s_csv, y_csv, y_w_csv, "Filtr Wienera")
                        rysuj_wykres(ax2, t_csv, s_csv, y_csv, y_sg_csv, "Filtr Savitzky-Golay")
                        rysuj_wykres(ax3, t_csv, s_csv, y_csv, y_e_csv, "Filtr EMD")
                        
                        st.pyplot(fig3)
                        
        except Exception as e:
            st.error(f"Błąd przetwarzania pliku: {e}")