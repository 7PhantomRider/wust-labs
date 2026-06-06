import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal
import emd

st.set_page_config(
    page_title="Lista 8 · Spektrogramy i EMD",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, code { font-family: 'IBM Plex Mono', monospace; }

.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #0a0e14; padding: 5px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.73rem;
    color: #556068; background: transparent; border-radius: 6px; padding: 7px 12px;
}
.stTabs [aria-selected="true"] {
    background: #141a21 !important; color: #acf2bd !important;
}

.info-box {
    background: #0a0e14; border-left: 3px solid #acf2bd;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    margin-bottom: 14px; font-size: 0.87rem; color: #8b949e; line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)

st.title("Lista 8 · Spektrogramy i Dekompozycja EMD")
st.caption("Przetwarzanie strumieni danych · Analiza Czasowo-Częstotliwościowa i HHT")

PLT = {
    "figure.facecolor": "#0a0e14", "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#21262d", "axes.labelcolor": "#8b949e",
    "xtick.color": "#556068", "ytick.color": "#556068",
    "grid.color": "#161b22", "text.color": "#8b949e",
    "axes.titlecolor": "#acf2bd", "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "grid.linestyle": "--", "grid.alpha": 0.5,
    "legend.facecolor": "#141a21", "legend.edgecolor": "#21262d",
    "legend.fontsize": 8,
}

CAT_COLORS = [
    "#acf2bd","#79c0ff","#d2a8ff","#ffa657","#ff7b72",
    "#56d364","#f0883e","#58a6ff","#bc8cff", "#ff7b72"
]

def generate_signal(typ, fs, dur, f0, f1=None, noise=0.0):
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    
    if typ == "Sinus": 
        x = np.sin(2 * np.pi * f0 * t)
    elif typ == "Prostokątny": 
        x = scipy_signal.square(2 * np.pi * f0 * t)
    elif typ == "Piłokształtny": 
        x = scipy_signal.sawtooth(2 * np.pi * f0 * t)
    elif typ == "Świergotliwy (Chirp)": 
        x = scipy_signal.chirp(t, f0=f0, f1=f1 if f1 else f0*2, t1=dur, method='linear')
    elif typ == "Mix (Superpozycja)": 
        x = np.sin(2 * np.pi * f0 * t) + 0.5 * np.cos(2 * np.pi * (f0 * 2.5) * t) + 0.2 * np.sin(2 * np.pi * (f0 * 5) * t)
    elif typ == "Impuls": 
        x = scipy_signal.unit_impulse(len(t), 'mid')
        
    x = x + noise * np.random.randn(len(t))
    return t, x

def get_spectrum(x, fs):
    """Szybkie liczenie widma amplitudowego (FFT) dla dodatnich częstotliwości."""
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1/fs)
    amp = np.abs(X) / N
    return freqs, amp

# EMD i widma
def plot_emd_and_spectra(t, x, imfs, fs, title_prefix):
    if imfs.ndim == 1:
        imfs = imfs.reshape(-1, 1)
        
    num_imfs = imfs.shape[1]
    
    with plt.rc_context(PLT):
        fig = plt.figure(figsize=(14, 2.5 * (num_imfs + 1)))
        fig.patch.set_facecolor("#0a0e14")
        gs = gridspec.GridSpec(num_imfs + 1, 2, figure=fig, hspace=0.6, wspace=0.2)
        
        # oryginał z widmem
        ax_orig_t = fig.add_subplot(gs[0, 0])
        ax_orig_t.plot(t, x, color="#e6edf3", lw=1.0)
        ax_orig_t.set_title(f"{title_prefix} - Sygnał Oryginalny")
        ax_orig_t.grid(True)
        
        ax_orig_f = fig.add_subplot(gs[0, 1])
        f_orig, amp_orig = get_spectrum(x, fs)
        ax_orig_f.plot(f_orig, amp_orig, color="#e6edf3", lw=1.0)
        ax_orig_f.set_title("Widmo Amplitudowe")
        ax_orig_f.set_xlim(0, fs/2)
        ax_orig_f.grid(True)
        
        # IMF i ich widma
        for i in range(num_imfs):
            col = CAT_COLORS[i % len(CAT_COLORS)]
            imf = imfs[:, i]
            
            # czas
            ax_t = fig.add_subplot(gs[i+1, 0])
            ax_t.plot(t, imf, color=col, lw=1.0)
            ax_t.set_title(f"IMF {i+1}")
            ax_t.grid(True)
            
            # częstotliwość
            ax_f = fig.add_subplot(gs[i+1, 1])
            f_imf, amp_imf = get_spectrum(imf, fs)
            ax_f.plot(f_imf, amp_imf, color=col, lw=1.2)
            ax_f.fill_between(f_imf, amp_imf, alpha=0.2, color=col)
            ax_f.set_title(f"Widmo IMF {i+1}")
            ax_f.set_xlim(0, fs/2)
            ax_f.grid(True)
            
            if i == num_imfs - 1:
                ax_t.set_xlabel("Czas [s]")
                ax_f.set_xlabel("Częstotliwość [Hz]")
                
    return fig

tabs = st.tabs([
    "Zad 1 · Spektrogramy",
    "Zad 2 & 3 · Dekompozycja EMD",
    "Zad 4 · EMD z pliku CSV",
])

# 1
with tabs[0]:
    st.markdown("#### 〰 Analiza czasowo-częstotliwościowa (STFT)")
    
    c1, c2, c3, c4 = st.columns(4)
    sig_type = c1.selectbox("Typ sygnału", 
                            ["Sinus", "Prostokątny", "Piłokształtny", "Świergotliwy (Chirp)", "Mix (Superpozycja)", "Impuls"], 
                            index=3)
    fs_z1 = c2.slider("Próbkowanie fs [Hz]", 500, 4000, 1000, step=100, key="z1_fs")
    nperseg = c3.select_slider("Rozmiar okna FFT (nperseg)", options=[32, 64, 128, 256, 512, 1024], value=256)
    noise_z1 = c4.slider("Szum", 0.0, 2.0, 0.0, step=0.1, key="z1_n")
    
    # ukryte parametry
    f0_z1 = 10
    f1_z1 = 200
    if sig_type == "Świergotliwy (Chirp)":
        f0_z1 = st.slider("F_start Chirp [Hz]", 1, 200, 10, key="z1_c_f0")
        f1_z1 = st.slider("F_end Chirp [Hz]", 10, fs_z1//2, 200, key="z1_c_f1")
    elif sig_type in ["Sinus", "Prostokątny", "Piłokształtny", "Mix (Superpozycja)"]:
        f0_z1 = st.slider("Częstotliwość bazowa [Hz]", 1, fs_z1//4, 20, key="z1_f0")

    t, x = generate_signal(sig_type, fs_z1, dur=2.0, f0=f0_z1, f1=f1_z1, noise=noise_z1)

    with plt.rc_context(PLT):
        fig, (ax_t, ax_spec) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [1, 2]})
        fig.patch.set_facecolor("#0a0e14")
        
        # przebieg czasowy
        ax_t.plot(t, x, color="#79c0ff", lw=1.0)
        ax_t.set_title(f"Przebieg czasowy: {sig_type}")
        ax_t.set_xlim(0, 2.0)
        ax_t.set_ylabel("Amplituda")
        ax_t.grid(True)
        
        # spektrogram
        f, t_spec, Sxx = scipy_signal.spectrogram(x, fs_z1, nperseg=nperseg, window='hann')
        # skala db
        Sxx_dB = 10 * np.log10(Sxx + 1e-10) 
        
        img = ax_spec.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='magma')
        ax_spec.set_title(f"Spektrogram (Okno Hanna, N={nperseg})")
        ax_spec.set_ylabel("Częstotliwość [Hz]")
        ax_spec.set_xlabel("Czas [s]")
        ax_spec.set_ylim(0, fs_z1/2)
        fig.colorbar(img, ax=ax_spec, label='Moc [dB]')
        
        plt.tight_layout()
    st.pyplot(fig)


# 2 i 3
with tabs[1]:
    st.markdown("#### Empiryczna Dekompozycja na Mody (EMD)")
    
    c1, c2, c3, c4 = st.columns(4)
    sig_type_emd = c1.selectbox("Typ sygnału do EMD", 
                                ["Świergotliwy (Chirp)", "Mix (Superpozycja)", "Sinus", "Prostokątny"], 
                                index=0, key="z2_sig")
    fs_z2 = c2.slider("Próbkowanie fs [Hz]", 500, 2000, 1000, step=100, key="z2_fs")
    noise_z2 = c3.slider("Dodaj szum", 0.0, 1.0, 0.0, step=0.1, key="z2_n")
    max_imfs = c4.slider("Max liczba IMF (zabezpieczenie)", 3, 10, 6, key="z2_max_imf")

    t_emd, x_emd = generate_signal(sig_type_emd, fs_z2, dur=1.0, f0=15, f1=150, noise=noise_z2)

    if st.button("Uruchom EMD (Może chwilę potrwać)", type="primary"):
        with st.spinner("Obliczanie przesiewania (sifting)..."):
            try:
                # wykonanie EMD
                imfs = emd.sift.sift(x_emd, max_imfs=max_imfs)
                
                # Rysowanie za pomocą naszej reużywalnej funkcji
                fig_emd = plot_emd_and_spectra(t_emd, x_emd, imfs, fs_z2, sig_type_emd)
                st.pyplot(fig_emd)
                
            except Exception as e:
                st.error(f"Błąd podczas obliczania EMD: {e}")
                

# 4
with tabs[2]:
    st.markdown("#### Dekompozycja EMD sygnału z pliku CSV")
    
    uploaded_file = st.file_uploader("Wybierz plik .csv z sygnałem", type="csv")
    
    if uploaded_file is not None:
        try:
            # opcje dla wczytania
            c1, c2, c3 = st.columns(3)
            sep = c1.text_input("Separator kolumn", value=",")
            fs_csv = c2.number_input("Częstotliwość próbkowania fs [Hz] (Podaj ręcznie)", min_value=1, value=500)
            max_imfs_csv = c3.slider("Max IMF dla pliku", 3, 10, 6, key="z4_max_imf")
            
            df = pd.read_csv(uploaded_file, sep=sep)
            
            st.write("Podgląd danych:")
            st.dataframe(df.head(3))
            
            # kolumna choice
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.error("Plik nie zawiera kolumn z danymi numerycznymi!")
            else:
                col_to_analyze = st.selectbox("Wybierz kolumnę z sygnałem do EMD:", numeric_cols)
                
                raw_data = df[col_to_analyze].dropna().values
                
                x_csv = np.ascontiguousarray(raw_data, dtype=np.float64).flatten()
                
                if np.std(x_csv) != 0:
                    x_csv = (x_csv - np.mean(x_csv)) / np.std(x_csv)
                
                
                x_csv = x_csv + np.random.randn(len(x_csv)) * 1e-6
                
                t_csv = np.arange(len(x_csv)) / fs_csv
                
                if st.button("Uruchom EMD dla pliku", key="btn_csv", type="primary"):
                    with st.spinner("Przetwarzanie pliku CSV... to może potrwać chwilę."):
                        try:
                            imfs_csv = emd.sift.sift(x_csv, max_imfs=max_imfs_csv)
                            
                            if imfs_csv.ndim == 1:
                                imfs_csv = imfs_csv.reshape(-1, 1)
                                
                            fig_csv = plot_emd_and_spectra(t_csv, x_csv, imfs_csv, fs_csv, f"Plik CSV: {uploaded_file.name}")
                            st.pyplot(fig_csv)
                        except Exception as e:
                            st.error(f"Sygnał nadal jest zbyt problematyczny dla EMD. Błąd: {e}")
                            
        except Exception as e:
            st.error(f"Wystąpił problem z wczytaniem pliku. Sprawdź separator. Błąd: {e}")
    else:
        st.info("Oczekuję na plik CSV.")