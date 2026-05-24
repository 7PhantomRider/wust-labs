"""
Lista 6 — Okna czasowe i FFT
Przetwarzanie strumieni danych
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import io

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lista 6 · Okna & FFT",
    page_icon="◈",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, code { font-family: 'JetBrains Mono', monospace; }

.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #111318; padding: 5px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
    color: #6e7681; background: transparent; border-radius: 6px; padding: 7px 13px;
}
.stTabs [aria-selected="true"] { background: #1c2128 !important; color: #e6edf3 !important; }

.info-box {
    background: #111318; border-left: 3px solid #79c0ff;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    margin-bottom: 14px; font-size: 0.87rem; color: #b1bac4; line-height: 1.65;
}
.warn-box {
    background: #1f1700; border-left: 3px solid #e3b341;
    padding: 10px 16px; border-radius: 0 8px 8px 0;
    margin-bottom: 14px; font-size: 0.87rem; color: #e3b341; line-height: 1.6;
}
.formula {
    background: #1c2128; border: 1px solid #30363d;
    padding: 7px 13px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    color: #79c0ff; margin: 6px 0 10px 0; display: inline-block;
}
.window-card {
    background: #1c2128; border: 1px solid #30363d;
    border-radius: 8px; padding: 10px 14px; margin: 4px 0;
    font-size: 0.83rem; color: #b1bac4;
}
</style>
""", unsafe_allow_html=True)

st.title("◈ Lista 6 · Okna czasowe i FFT")
st.caption("Przetwarzanie strumieni danych · Windowing & analiza widmowa")

# ─── Helpers ──────────────────────────────────────────────────────────────────
PLT = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#111318",
    "axes.edgecolor": "#21262d", "axes.labelcolor": "#b1bac4",
    "xtick.color": "#6e7681", "ytick.color": "#6e7681",
    "grid.color": "#1c2128", "text.color": "#b1bac4",
    "axes.titlecolor": "#79c0ff", "axes.titlesize": 10,
    "axes.labelsize": 8.5, "grid.linestyle": "--", "grid.alpha": 0.6,
    "legend.facecolor": "#1c2128", "legend.edgecolor": "#30363d",
    "legend.fontsize": 8,
}

WINDOWS = {
    "Dirichleta (prostokąt)": ("boxcar",   "#ff7b72"),
    "Hanna":                  ("hann",     "#79c0ff"),
    "Hamminga":               ("hamming",  "#3fb950"),
    "Blackmana":              ("blackman", "#d2a8ff"),
}

WIN_KEYS   = list(WINDOWS.keys())
WIN_COLORS = {k: v[1] for k, v in WINDOWS.items()}
WIN_IDS    = {k: v[0] for k, v in WINDOWS.items()}

SIGNAL_TYPES = ["Sinus", "Cosinus", "Prostokątny", "Piłokształtny", "Suma 3 sinusów", "Szum biały"]


def make_window(name, N):
    return signal.get_window(WIN_IDS[name], N)


def make_signal(sig_type, t, f0):
    if sig_type == "Sinus":
        return np.sin(2 * np.pi * f0 * t)
    elif sig_type == "Cosinus":
        return np.cos(2 * np.pi * f0 * t)
    elif sig_type == "Prostokątny":
        return signal.square(2 * np.pi * f0 * t)
    elif sig_type == "Piłokształtny":
        return signal.sawtooth(2 * np.pi * f0 * t)
    elif sig_type == "Suma 3 sinusów":
        return (np.sin(2*np.pi*f0*t) +
                0.5*np.sin(2*np.pi*2*f0*t) +
                0.25*np.sin(2*np.pi*3*f0*t))
    elif sig_type == "Szum biały":
        rng = np.random.default_rng(42)
        return rng.standard_normal(len(t))
    return np.sin(2 * np.pi * f0 * t)


def compute_spectrum(x, fs, window_name=None, db=False):
    """Zwraca (freqs, amp) dla jednostronnego widma z opcjonalnym oknem."""
    N = len(x)
    if window_name:
        w = make_window(window_name, N)
        # Normalizacja okna żeby amplitudy były porównywalne
        x = x * w / np.mean(w)
    X = fft(x)
    freqs = fftfreq(N, d=1/fs)
    half = N // 2
    amp = (2 / N) * np.abs(X[:half])
    freqs = freqs[:half]
    if db:
        amp = 20 * np.log10(amp + 1e-12)
    return freqs, amp


# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "Zad 1 · Kształty okien",
    "Zad 2 · Okna na sygnale",
    "Zad 3 · FFT z wyborem sygnału",
    "Zad 4 · Ładowanie z CSV",
])

# ══════════════════════════════════════════════════════════════════════════════
# ZAD 1 — Kształty okien i ich widma
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    c1, c2 = st.columns([2, 1])
    N = c1.slider("Liczba próbek okna N", 32, 1024, 256, 32, key="z1n")
    scale = c2.radio("Skala widma", ["Liniowa", "Decybelowa (dB)"], horizontal=True, key="z1sc")
    use_db = scale == "Decybelowa (dB)"

    with plt.rc_context(PLT):
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor("#0d1117")
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.85, wspace=0.35)

        for i, (name, (wid, col)) in enumerate(WINDOWS.items()):
            w = make_window(name, N)
            n = np.arange(N)

            # Widmo okna
            W = fft(w, n=4096)
            freqs_w = np.linspace(-0.5, 0.5, 4096)
            amp_w = np.abs(W) / np.max(np.abs(W))
            if use_db:
                amp_w = 20 * np.log10(amp_w + 1e-12)

            ax_t = fig.add_subplot(gs[i, 0])
            ax_t.plot(n, w, color=col, lw=1.4)
            ax_t.fill_between(n, w, alpha=0.15, color=col)
            ax_t.set_title(f"{name} — kształt")
            ax_t.set_xlabel("Próbka n"); ax_t.set_ylabel("w[n]")
            ax_t.set_ylim(-0.1, 1.15); ax_t.grid(True)

            ax_f = fig.add_subplot(gs[i, 1])
            ax_f.plot(freqs_w, amp_w, color=col, lw=1.2)
            ax_f.set_title(f"{name} — widmo okna")
            ax_f.set_xlabel("Częstotliwość znormalizowana")
            ax_f.set_ylabel("dB" if use_db else "|W(f)| norm.")
            ax_f.set_xlim(-0.5, 0.5)
            if use_db:
                ax_f.set_ylim(-120, 5)
                ax_f.axhline(-60, color="#30363d", lw=0.7, ls=":")
            ax_f.grid(True)

    st.pyplot(fig)


    with st.expander("📖 Parametry okien — tabela porównawcza"):
        st.markdown("""
        | Okno | Szerokość piku głównego | Tłumienie listków bocznych | Kiedy używać |
        |---|---|---|---|
        | **Dirichleta** | 2 próbki | ~−13 dB | Gdy sygnał jest dokładnie periodyczny w oknie |
        | **Hanna** | 4 próbki | ~−32 dB | Ogólny wybór, dobry kompromis |
        | **Hamminga** | 4 próbki | ~−43 dB | Gdy ważna jest pierwsza listka boczna |
        | **Blackmana** | 6 próbek | ~−58 dB | Gdy potrzebna jest maksymalna izolacja |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# ZAD 2 — Widmo sinusa przy różnych oknach (3 częstotliwości)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:

    # Suwaki: typ sygnału + 3 częstotliwości
    st.markdown("#### ⚙️ Parametry sygnału")
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    sig_type = cc1.selectbox("Typ sygnału", SIGNAL_TYPES, key="z2sig")
    f1 = cc2.slider("f₁ [Hz]", 1, 200, 10,  key="z2f1")
    f2 = cc3.slider("f₂ [Hz]", 1, 200, 50,  key="z2f2")
    f3 = cc4.slider("f₃ [Hz]", 1, 200, 120, key="z2f3")
    fs = cc5.slider("fs [Hz]", 500, 8000, 1000, 100, key="z2fs")

    cw1, cw2 = st.columns(2)
    show_db = cw1.checkbox("Skala dB", value=False, key="z2db")
    show_original = cw2.checkbox("Pokaż sygnał bez okna (Dirichleta) jako tło", value=True, key="z2bg")

    # Wybór okien do porównania
    sel_windows = st.multiselect(
        "Które okna porównywać?",
        WIN_KEYS,
        default=WIN_KEYS,
        key="z2win"
    )

    dur = 1.0
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)

    # Budujemy sygnał z 3 częstotliwościami
    s1 = make_signal(sig_type, t, f1)
    s2 = make_signal(sig_type, t, f2) * 0.6
    s3 = make_signal(sig_type, t, f3) * 0.3
    x  = s1 + s2 + s3

    with plt.rc_context(PLT):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117")

        # Sygnał w czasie
        display_samples = min(int(fs * 0.15), len(t))
        axes[0].plot(t[:display_samples], x[:display_samples],
                     color="#e6edf3", lw=0.9, alpha=0.85)
        axes[0].set_title(f"Sygnał: {sig_type} (f₁={f1}, f₂={f2}, f₃={f3} Hz)")
        axes[0].set_xlabel("Czas [s]"); axes[0].set_ylabel("Amplituda")
        axes[0].grid(True)

        # Widma z różnymi oknami
        if show_original and "Dirichleta (prostokąt)" not in sel_windows:
            fr_bg, amp_bg = compute_spectrum(x, fs, "Dirichleta (prostokąt)", db=show_db)
            axes[1].plot(fr_bg, amp_bg, color="#30363d", lw=1.0, alpha=0.5,
                         ls="--", label="Dirichleta (tło)")

        for wname in sel_windows:
            fr, amp = compute_spectrum(x, fs, wname, db=show_db)
            axes[1].plot(fr, amp, color=WIN_COLORS[wname], lw=1.3,
                         alpha=0.85, label=wname)

        axes[1].set_title("Widma amplitudowe przy różnych oknach")
        axes[1].set_xlabel("Częstotliwość [Hz]")
        axes[1].set_ylabel("dB" if show_db else "|X(f)|")
        axes[1].set_xlim(0, min(fs/2, max(f1,f2,f3)*2.5 + 30))
        if show_db:
            axes[1].set_ylim(-100, 10)
        axes[1].legend(); axes[1].grid(True)

        # Oznaczenie pików
        for fx, lbl in [(f1,"f₁"), (f2,"f₂"), (f3,"f₃")]:
            axes[1].axvline(fx, color="#e3b341", lw=0.7, ls=":", alpha=0.6)
            axes[1].text(fx+1, axes[1].get_ylim()[1]*0.92, lbl,
                         color="#e3b341", fontsize=8, family="monospace")

        plt.tight_layout()
    st.pyplot(fig)

    st.info("💡 Ustaw f₁=10.5 Hz i fs=1000 Hz, N=1000 próbek → f₁ nie trafia dokładnie w prążek FFT. "
            "Widać przeciek przy Dirichleta. Zmień na Blackmana — przeciek zostaje stłumiony.")

# ══════════════════════════════════════════════════════════════════════════════
# ZAD 3 — FFT z pełnym wyborem sygnału i okna (scipy.fft)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:

    c1, c2, c3, c4, c5 = st.columns(5)
    sig_type3 = c1.selectbox("Typ sygnału", SIGNAL_TYPES, key="z3sig")
    f0_3      = c2.slider("Częstotliwość f₀ [Hz]", 1, 300, 25, key="z3f0")
    fs_3      = c3.slider("fs [Hz]", 200, 8000, 1000, 100, key="z3fs")
    dur_3     = c4.slider("Czas [s]", 0.1, 5.0, 1.0, 0.1, key="z3d")
    win_3     = c5.selectbox("Okno", WIN_KEYS, index=1, key="z3w")

    sc1, sc2, sc3 = st.columns(3)
    show_phase  = sc1.checkbox("Pokaż widmo fazowe", value=True, key="z3ph")
    show_db3    = sc2.checkbox("Skala dB", value=False, key="z3db")
    zero_pad    = sc3.checkbox("Zero-padding ×4 (więcej punktów FFT)", value=False, key="z3zp")

    t3 = np.linspace(0, dur_3, int(fs_3 * dur_3), endpoint=False)
    x3 = make_signal(sig_type3, t3, f0_3)

    N3 = len(x3)
    w3 = make_window(win_3, N3)
    x3w = x3 * w3 / np.mean(w3)

    fft_N = N3 * 4 if zero_pad else N3
    X3 = fft(x3w, n=fft_N)
    freqs3 = fftfreq(fft_N, d=1/fs_3)
    half3 = fft_N // 2
    freqs3 = freqs3[:half3]
    amp3   = (2 / N3) * np.abs(X3[:half3])
    pha3   = np.angle(X3[:half3], deg=True)
    if show_db3:
        amp3 = 20 * np.log10(amp3 + 1e-12)

    ncols = 3 if show_phase else 2
    with plt.rc_context(PLT):
        fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4.5))
        fig.patch.set_facecolor("#0d1117")
        if ncols == 2:
            axes = list(axes)

        disp = min(int(fs_3 * 0.12), len(t3))
        axes[0].plot(t3[:disp], x3[:disp], color="#e6edf3", lw=1.0, alpha=0.6, label="oryg.")
        axes[0].plot(t3[:disp], x3w[:disp], color="#79c0ff", lw=1.2, label=f"× okno {win_3[:8]}")
        axes[0].set_title("Sygnał × okno")
        axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("Amplituda")
        axes[0].legend(); axes[0].grid(True)

        axes[1].plot(freqs3, amp3, color="#3fb950", lw=1.1)
        axes[1].set_title(f"Widmo amplitudowe · scipy.fft{' (dB)' if show_db3 else ''}")
        axes[1].set_xlabel("f [Hz]")
        axes[1].set_ylabel("dB" if show_db3 else "|X(f)|")
        axes[1].set_xlim(0, min(fs_3/2, f0_3 * 5 + 50))
        axes[1].grid(True)
        axes[1].axvline(f0_3, color="#e3b341", lw=0.8, ls=":", alpha=0.7)

        if show_phase:
            axes[2].plot(freqs3, pha3, color="#d2a8ff", lw=0.9, alpha=0.8)
            axes[2].set_title("Widmo fazowe ∠X(f) [°]")
            axes[2].set_xlabel("f [Hz]"); axes[2].set_ylabel("Faza [°]")
            axes[2].set_xlim(0, min(fs_3/2, f0_3 * 5 + 50))
            axes[2].set_ylim(-200, 200); axes[2].grid(True)

        plt.tight_layout()
    st.pyplot(fig)

    if zero_pad:
        st.info(f"💡 Zero-padding ×4: zamiast {N3} punktów FFT masz {fft_N}. "
                "Widmo jest bardziej 'gładkie' i pikuje dokładniej — ale nie dodaje nowej informacji! "
                "To interpolacja widma, nie większa rozdzielczość.")

# ══════════════════════════════════════════════════════════════════════════════
# ZAD 4 — FFT z pliku CSV
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:

    # ── sekcja ładowania ──────────────────────────────────────────────────────
    load_mode = st.radio(
        "Źródło danych",
        ["📂 Wgraj własny plik CSV", "📋 Użyj wbudowanego przykładu"],
        horizontal=True, key="z4mode"
    )

    df = None
    fs_csv = 1000
    col_sig = None
    col_time = None

    if load_mode == "📋 Użyj wbudowanego przykładu":
       
        try:
            df = pd.read_csv("sample_signal.csv")
            col_time = "time"
            col_sig  = "signal"
            # Wyznacz fs z kolumny czasu
            dt = df[col_time].iloc[1] - df[col_time].iloc[0]
            fs_csv = int(round(1.0 / dt))
            st.success(f"✓ Załadowano {len(df)} próbek · fs={fs_csv} Hz")
        except FileNotFoundError:
            st.error("Nie znaleziono pliku sample_signal.csv. Upewnij się że jest w tym samym folderze co skrypt.")
            st.stop()

    else:  # upload własnego pliku
        uploaded = st.file_uploader(
            "Wgraj plik CSV",
            type=["csv", "txt"],
            help="Plik powinien mieć nagłówki. Minimalne wymaganie: kolumna z sygnałem."
        )

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✓ Załadowano: {uploaded.name} · {len(df)} wierszy · kolumny: {list(df.columns)}")
            except Exception as e:
                st.error(f"Błąd odczytu: {e}")
                st.stop()
        else:
            st.info("⬆️ Wgraj plik CSV lub wybierz przykład po lewej.")
            st.markdown("""
            **Oczekiwany format:**
            ```
            time,signal
            0.000,0.234
            0.001,0.456
            ...
            ```
            lub tylko kolumna signal (wtedy podaj fs ręcznie poniżej).
            """)

    if df is not None:
        # ── wybór kolumn ──────────────────────────────────────────────────────
        st.markdown("#### ⚙️ Konfiguracja")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        ca, cb, cc = st.columns(3)
        col_sig_sel = ca.selectbox(
            "Kolumna: sygnał", num_cols,
            index=num_cols.index(col_sig) if col_sig in num_cols else 0,
            key="z4cs"
        )
        time_options = ["— brak (użyj fs) —"] + num_cols
        col_time_sel = cb.selectbox(
            "Kolumna: czas (opcjonalnie)", time_options,
            index=time_options.index(col_time) if col_time in time_options else 0,
            key="z4ct"
        )

        # Wyznacz fs
        if col_time_sel != "— brak (użyj fs) —":
            try:
                dt_auto = df[col_time_sel].iloc[1] - df[col_time_sel].iloc[0]
                fs_auto = int(round(1.0 / dt_auto))
                fs_csv = cc.number_input("fs [Hz] (wyznaczone z czasu)", value=fs_auto,
                                         min_value=1, key="z4fs_auto")
            except:
                fs_csv = cc.number_input("fs [Hz]", value=1000, min_value=1, key="z4fs_man")
        else:
            fs_csv = cc.number_input("fs [Hz] (podaj ręcznie)", value=1000,
                                     min_value=1, key="z4fs_man2")

        # ── opcje analizy ─────────────────────────────────────────────────────
        cd, ce, cf = st.columns(3)
        win_csv  = cd.selectbox("Okno", WIN_KEYS, index=1, key="z4win")
        db_csv   = ce.checkbox("Skala dB", value=False, key="z4db")
        show_sg  = cf.checkbox("Pokaż spektrogram", value=True, key="z4sg")

        x_csv = df[col_sig_sel].dropna().values.astype(float)

        # ── wykresy ───────────────────────────────────────────────────────────
        t_csv = np.arange(len(x_csv)) / fs_csv

        fr_csv, amp_csv = compute_spectrum(x_csv, fs_csv, win_csv, db=db_csv)

        ncols4 = 3 if show_sg else 2
        with plt.rc_context(PLT):
            fig, axes = plt.subplots(1, ncols4, figsize=(5*ncols4, 4.5))
            fig.patch.set_facecolor("#0d1117")

            # Sygnał
            max_disp = min(len(t_csv), int(fs_csv * 0.5))
            axes[0].plot(t_csv[:max_disp], x_csv[:max_disp],
                         color="#79c0ff", lw=0.9)
            axes[0].set_title(f"Sygnał z CSV: {col_sig_sel!r}")
            axes[0].set_xlabel("Czas [s]"); axes[0].set_ylabel("Amplituda")
            axes[0].grid(True)

            # Widmo
            axes[1].plot(fr_csv, amp_csv, color="#3fb950", lw=1.1)
            axes[1].set_title(f"Widmo amplitudowe (okno: {win_csv[:10]})")
            axes[1].set_xlabel("f [Hz]")
            axes[1].set_ylabel("dB" if db_csv else "|X(f)|")
            axes[1].set_xlim(0, fs_csv / 2)
            axes[1].grid(True)

            # Spektrogram
            if show_sg:
                nperseg = min(256, len(x_csv) // 4)
                f_sg, t_sg, Sxx = signal.spectrogram(x_csv, fs_csv, nperseg=nperseg)
                axes[2].pcolormesh(t_sg, f_sg,
                                   10*np.log10(Sxx + 1e-12),
                                   shading='gouraud', cmap='magma')
                axes[2].set_title("Spektrogram (czas × częstotliwość)")
                axes[2].set_xlabel("Czas [s]"); axes[2].set_ylabel("f [Hz]")

            plt.tight_layout()
        st.pyplot(fig)

        # ── podsumowanie ──────────────────────────────────────────────────────
        with st.expander("📊 Statystyki sygnału"):
            peak_idx = np.argmax(amp_csv)
            st.markdown(f"""
            | Parametr | Wartość |
            |---|---|
            | Liczba próbek | {len(x_csv)} |
            | Czas trwania | {len(x_csv)/fs_csv:.3f} s |
            | Częstotliwość próbkowania | {fs_csv} Hz |
            | Dominująca częstotliwość | **{fr_csv[peak_idx]:.1f} Hz** |
            | Wartość RMS | {np.sqrt(np.mean(x_csv**2)):.4f} |
            | Min / Max | {x_csv.min():.4f} / {x_csv.max():.4f} |
            """)

        # ── eksport widma do CSV ──────────────────────────────────────────────
        df_out = pd.DataFrame({"frequency_hz": fr_csv, "amplitude": amp_csv})
        csv_bytes = df_out.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Pobierz widmo jako CSV",
            data=csv_bytes,
            file_name="spectrum_output.csv",
            mime="text/csv",
        )