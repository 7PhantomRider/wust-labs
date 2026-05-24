"""
Lista 7 — Falki (Wavelets)
Przetwarzanie strumieni danych
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal
import pywt

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lista 7 · Falki",
    page_icon="〰",
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
.warn-box {
    background: #160f00; border-left: 3px solid #e3b341;
    padding: 10px 16px; border-radius: 0 8px 8px 0;
    margin-bottom: 12px; font-size: 0.86rem; color: #c9a227; line-height: 1.6;
}
.formula {
    background: #141a21; border: 1px solid #21262d;
    padding: 7px 13px; border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
    color: #acf2bd; margin: 6px 0 10px 0; display: inline-block;
}
.wavelet-badge {
    display: inline-block; background: #141a21; border: 1px solid #21262d;
    border-radius: 4px; padding: 2px 8px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; color: #79c0ff; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

st.title("〰 Lista 7 · Falki (Wavelets)")
st.caption("Przetwarzanie strumieni danych · Analiza falkowa")

# ─── Helpers ──────────────────────────────────────────────────────────────────
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

# Katalog falek: (nazwa_wyświetlana, pywt_name, typ, opis)
WAVELET_CATALOG = [
    ("Haar",               "haar",    "discrete", "Najprostsza — prostokąt. db1. Skokowa, ale super szybka."),
    ("Daubechies db2",     "db2",     "discrete", "Gładka, kompaktowa. Dobra do sygnałów nieciągłych."),
    ("Daubechies db4",     "db4",     "discrete", "Większa gładkość, dłuższe nośniki."),
    ("Symlets sym4",       "sym4",    "discrete", "Quasi-symetryczna wersja Daubechies."),
    ("Coiflets coif3",     "coif3",   "discrete", "Jeszcze bardziej symetryczna, dobre momenty znikające."),
    ("Biortogonalna bior2.2","bior2.2","discrete", "Para falek analizująca i rekonstruująca. Używana w JPEG2000."),
    ("Gaussian gaus2",     "gaus2",   "continuous","Pochodna Gaussa. Wygładzona, dobra do detekcji krawędzi."),
    ("Meksykański kapelusz","mexh",   "continuous","Druga pochodna Gaussa. Klasyczny kształt 'hat'."),
    ("Morlet",             "morl",    "continuous","Kompleksowy sinus modulowany Gaussem. Król spektrogramów."),
]

CAT_COLORS = [
    "#acf2bd","#79c0ff","#d2a8ff","#ffa657","#ff7b72",
    "#56d364","#f0883e","#58a6ff","#bc8cff"
]

def get_wavelet_data(pywt_name, wav_type, n_points=512):
    """Pobiera dane falki macierzystej (phi, psi) lub tylko psi dla ciągłych."""
    try:
        if wav_type == "discrete":
            w = pywt.Wavelet(pywt_name)
            phi, psi, x = w.wavefun(level=8)
            return x, phi, psi
        else:  # continuous
            w = pywt.ContinuousWavelet(pywt_name)
            psi, x = w.wavefun(length=n_points)
            return x, None, np.real(psi)
    except Exception:
        return None, None, None

def make_chirp(fs=2000, dur=2.0, f0=5, f1=500):
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    x = scipy_signal.chirp(t, f0=f0, f1=f1, t1=dur, method='linear')
    return t, x

# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "Zad 1 · Galeria falek",
    "Zad 2 · Daubechies dbN",
    "Zad 3 · Dekompozycja chirp",
])

# ══════════════════════════════════════════════════════════════════════════════
# ZAD 1 — Galeria wszystkich typów falek
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    # Filtry wyboru
    c1, c2 = st.columns([3, 1])
    show_types = c1.multiselect(
        "Pokaż typy falek",
        ["discrete", "continuous"],
        default=["discrete", "continuous"],
        format_func=lambda x: "Dyskretne (DWT)" if x == "discrete" else "Ciągłe (CWT)",
        key="z1types"
    )
    n_pts = c2.slider("Punktów falki", 128, 1024, 512, 128, key="z1pts")

    filtered = [(name, pwn, wt, desc, col)
                for (name, pwn, wt, desc), col
                in zip(WAVELET_CATALOG, CAT_COLORS)
                if wt in show_types]

    if not filtered:
        st.warning("Wybierz przynajmniej jeden typ falki.")
    else:
        with plt.rc_context(PLT):
            rows = len(filtered)
            fig = plt.figure(figsize=(13, 2.8 * rows))
            fig.patch.set_facecolor("#0a0e14")
            gs = gridspec.GridSpec(rows, 3, figure=fig, hspace=0.9, wspace=0.35)

            for i, (name, pwn, wt, desc, col) in enumerate(filtered):
                x, phi, psi = get_wavelet_data(pwn, wt, n_pts)
                if x is None:
                    continue

                # Kolumna 0: φ(t) lub pusta dla ciągłych
                ax0 = fig.add_subplot(gs[i, 0])
                if phi is not None and len(phi) > 0:
                    ax0.plot(x, phi, color=col, lw=1.3)
                    ax0.fill_between(x, phi, alpha=0.12, color=col)
                    ax0.set_title(f"φ(t) — scaling")
                else:
                    ax0.text(0.5, 0.5, "brak φ(t)\n(falka ciągła)",
                             ha='center', va='center',
                             transform=ax0.transAxes, color="#556068", fontsize=8)
                    ax0.set_title("φ(t)")
                ax0.set_ylabel(name[:14], fontsize=8, color=col)
                ax0.grid(True); ax0.set_xlabel("t")

                # Kolumna 1: ψ(t) — falka macierzysta
                ax1 = fig.add_subplot(gs[i, 1])
                ax1.plot(x, psi, color=col, lw=1.4)
                ax1.fill_between(x, psi, alpha=0.12, color=col)
                ax1.set_title(f"ψ(t) — mother wavelet · {name}")
                ax1.grid(True); ax1.set_xlabel("t")

                # Kolumna 2: widmo |FFT(ψ)|
                ax2 = fig.add_subplot(gs[i, 2])
                N_fft = len(psi)
                Psi = np.fft.fft(psi, n=N_fft*4)
                freqs = np.fft.fftfreq(N_fft*4)
                half = N_fft*2
                amp = np.abs(Psi[:half])
                amp /= (amp.max() + 1e-12)
                ax2.plot(freqs[:half], amp, color=col, lw=1.2)
                ax2.fill_between(freqs[:half], amp, alpha=0.1, color=col)
                ax2.set_title("|FFT(ψ)| — pasmo falki")
                ax2.set_xlim(0, 0.5); ax2.grid(True); ax2.set_xlabel("f znorm.")

        st.pyplot(fig)

        with st.expander("📖 Kiedy używać której falki?"):
            st.markdown("""
            | Falka | Zastosowanie |
            |---|---|
            | **Haar** | Detekcja kroków, skoków, bardzo szybka analiza |
            | **Daubechies dbN** | Ogólne zastosowania — im wyższe N, tym gładszy sygnał |
            | **Symlets** | Podobnie do db, ale lepsza symetria fazowa |
            | **Coiflets** | Gdy potrzeba "zerowych momentów" zarówno ψ jak i φ |
            | **Biortogonalna** | Kompresja obrazów (JPEG2000), możliwość liniowej fazy |
            | **Gaussian / Mexh** | Detekcja krawędzi, analiza czasu-częstotliwości (CWT) |
            | **Morlet** | Spektrogramy falkowe, analiza EEG, sejsmologia |
            """)

# ══════════════════════════════════════════════════════════════════════════════
# ZAD 2 — Daubechies w różnych wersjach
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:

    c1, c2, c3 = st.columns(3)
    db_min  = c1.slider("db od",  1, 15,  1, key="z2min")
    db_max  = c2.slider("db do",  2, 38, 10, key="z2max")
    db_view = c3.radio("Widok", ["ψ(t) wszystkie", "φ(t) i ψ(t) wybranej"], key="z2view")

    if db_min >= db_max:
        st.warning("'db od' musi być mniejsze niż 'db do'.")
    else:
        db_list = list(range(db_min, db_max + 1))
        palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(db_list)))

        if db_view == "ψ(t) wszystkie":
            # Wszystkie ψ(t) na jednym wykresie + osobno ich widma
            with plt.rc_context(PLT):
                fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(13, 5))
                fig.patch.set_facecolor("#0a0e14")

                for j, n in enumerate(db_list):
                    try:
                        w = pywt.Wavelet(f"db{n}")
                        phi, psi, x = w.wavefun(level=8)
                        col = palette[j]
                        ax_t.plot(x, psi, color=col, lw=1.1, label=f"db{n}", alpha=0.85)

                        Psi = np.fft.fft(psi, n=len(psi)*4)
                        freqs_db = np.fft.fftfreq(len(psi)*4)
                        half_db = len(psi)*2
                        amp_db = np.abs(Psi[:half_db])
                        amp_db /= (amp_db.max() + 1e-12)
                        ax_f.plot(freqs_db[:half_db], amp_db, color=col, lw=1.0,
                                  label=f"db{n}", alpha=0.8)
                    except Exception:
                        pass

                ax_t.set_title(f"Falki ψ(t): db{db_min} → db{db_max}")
                ax_t.set_xlabel("t"); ax_t.set_ylabel("ψ(t)"); ax_t.grid(True)
                if len(db_list) <= 12:
                    ax_t.legend(ncol=2, fontsize=7)

                ax_f.set_title("Widmo |FFT(ψ)| — wszystkie Daubechies")
                ax_f.set_xlabel("f znorm."); ax_f.set_xlim(0, 0.5); ax_f.grid(True)
                if len(db_list) <= 12:
                    ax_f.legend(ncol=2, fontsize=7)

                plt.tight_layout()
            st.pyplot(fig)

            # Długość filtra vs N
            with plt.rc_context(PLT):
                fig2, ax = plt.subplots(figsize=(8, 3))
                fig2.patch.set_facecolor("#0a0e14")
                lengths = []
                for n in db_list:
                    try:
                        w = pywt.Wavelet(f"db{n}")
                        lengths.append(len(w.dec_lo))
                    except:
                        lengths.append(0)
                ax.bar(db_list, lengths, color="#acf2bd", alpha=0.7, width=0.6)
                ax.set_title("Długość filtra vs rząd N")
                ax.set_xlabel("N (rząd Daubechies)"); ax.set_ylabel("Długość filtra")
                ax.grid(True, axis='y')
                plt.tight_layout()
            st.pyplot(fig2)

        else:
            # φ i ψ wybranej db
            db_sel = st.slider("Wybierz dbN do podglądu", db_min, db_max,
                               min(db_min + 2, db_max), key="z2sel")
            try:
                w = pywt.Wavelet(f"db{db_sel}")
                phi, psi, x = w.wavefun(level=8)

                with plt.rc_context(PLT):
                    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
                    fig.patch.set_facecolor("#0a0e14")

                    axes[0].plot(x, phi, color="#79c0ff", lw=1.4)
                    axes[0].fill_between(x, phi, alpha=0.15, color="#79c0ff")
                    axes[0].set_title(f"db{db_sel} — φ(t) scaling function")
                    axes[0].set_xlabel("t"); axes[0].grid(True)

                    axes[1].plot(x, psi, color="#acf2bd", lw=1.4)
                    axes[1].fill_between(x, psi, alpha=0.15, color="#acf2bd")
                    axes[1].set_title(f"db{db_sel} — ψ(t) mother wavelet")
                    axes[1].set_xlabel("t"); axes[1].grid(True)

                    # Filtry (współczynniki)
                    axes[2].stem(w.dec_lo, linefmt="#ffa657", markerfmt="o",
                                 basefmt=" ", label="dec_lo (φ)")
                    axes[2].stem(w.dec_hi, linefmt="#ff7b72", markerfmt="s",
                                 basefmt=" ", label="dec_hi (ψ)")
                    axes[2].set_title(f"db{db_sel} — współczynniki filtra")
                    axes[2].legend(); axes[2].grid(True)
                    axes[2].set_xlabel("k")

                    plt.tight_layout()
                st.pyplot(fig)

                st.markdown(f"""
                **db{db_sel} — właściwości:**
                | Parametr | Wartość |
                |---|---|
                | Długość filtra | {len(w.dec_lo)} |
                | Nośnik ψ(t) | [{x[0]:.1f}, {x[-1]:.1f}] |
                | Zerowe momenty | {db_sel} |
                | Symetria | {'tak' if w.symmetry == 'symmetric' else 'brak / asymetryczna'} |
                """)
            except Exception as e:
                st.error(f"Błąd: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ZAD 3 — Dekompozycja sygnału chirp
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
  
    # ── Parametry sygnału ────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Parametry sygnału")
    sc1, sc2, sc3, sc4 = st.columns(4)
    fs_c   = sc1.slider("fs [Hz]", 500, 8000, 2000, 100, key="z3fs")
    dur_c  = sc2.slider("Czas [s]", 0.5, 4.0, 2.0, 0.5, key="z3dur")
    f0_c   = sc3.slider("f_start [Hz]", 1, 200, 5,   key="z3f0")
    f1_c   = sc4.slider("f_end [Hz]",  10, 1000, min(int(fs_c*0.4), 500), key="z3f1")

    t_c, x_c = make_chirp(fs_c, dur_c, f0_c, f1_c)

    # ── Trzy falki do porównania ─────────────────────────────────────────────
    st.markdown("#### 〰 Wybór falek do porównania")

    DISCRETE_WAVELETS = [name for name in pywt.wavelist(kind='discrete')]

    wc1, wc2, wc3, wc4 = st.columns(4)
    wav1 = wc1.selectbox("Falka 1", DISCRETE_WAVELETS,
                         index=DISCRETE_WAVELETS.index("haar"), key="z3w1")
    wav2 = wc2.selectbox("Falka 2", DISCRETE_WAVELETS,
                         index=DISCRETE_WAVELETS.index("db4"), key="z3w2")
    wav3 = wc3.selectbox("Falka 3", DISCRETE_WAVELETS,
                         index=DISCRETE_WAVELETS.index("sym4"), key="z3w3")

    max_lvl = pywt.dwt_max_level(len(x_c), pywt.Wavelet(wav1).dec_len)
    max_lvl = min(max_lvl, 10)
    level = wc4.slider("Poziom dekompozycji", 1, max_lvl,
                       min(5, max_lvl), key="z3lvl")

    # ── Wizualizacja ─────────────────────────────────────────────────────────
    WAV_COLORS = ["#acf2bd", "#79c0ff", "#d2a8ff"]
    wav_list   = [wav1, wav2, wav3]

    with plt.rc_context(PLT):
        fig = plt.figure(figsize=(15, 3.5 + 2.2 * level))
        fig.patch.set_facecolor("#0a0e14")

        # Liczba wierszy: 1 (sygnał) + level+1 (cA + każdy cD) razy 3 kolumny
        n_rows = 1 + level + 1
        gs = gridspec.GridSpec(n_rows, 3, figure=fig,
                               hspace=0.9, wspace=0.32)

        # Wiersz 0: oryginalny sygnał (rozciągnięty przez 3 kolumny)
        ax_orig = fig.add_subplot(gs[0, :])
        ax_orig.plot(t_c, x_c, color="#e6edf3", lw=0.7, alpha=0.8)
        ax_orig.set_title(f"Chirp: {f0_c}→{f1_c} Hz, fs={fs_c} Hz, {dur_c}s")
        ax_orig.set_xlabel("Czas [s]"); ax_orig.set_ylabel("x(t)")
        ax_orig.grid(True)

        # Dla każdej z 3 falek: kolumna
        for col_idx, (wname, wcol) in enumerate(zip(wav_list, WAV_COLORS)):
            try:
                coeffs = pywt.wavedec(x_c, wname, level=level)
                # coeffs[0] = cA_level, coeffs[1] = cD_level, ..., coeffs[-1] = cD1

                # cA (approximation) — ostatni poziom aproksymacji
                cA = coeffs[0]
                t_cA = np.linspace(0, dur_c, len(cA))
                ax = fig.add_subplot(gs[1, col_idx])
                ax.plot(t_cA, cA, color=wcol, lw=1.1)
                ax.fill_between(t_cA, cA, alpha=0.1, color=wcol)
                ax.set_title(f"{wname} · cA{level} (approx)")
                ax.set_ylabel("Amplituda", fontsize=7.5)
                ax.grid(True)
                if col_idx == 0:
                    ax.set_ylabel(f"cA{level}", fontsize=8, color=wcol)

                # cD od poziomu level do 1
                for lvl_idx, cD in enumerate(coeffs[1:], start=1):
                    actual_lvl = level - lvl_idx + 1
                    t_cD = np.linspace(0, dur_c, len(cD))
                    row = 1 + lvl_idx
                    ax2 = fig.add_subplot(gs[row, col_idx])
                    ax2.plot(t_cD, cD, color=wcol, lw=0.9, alpha=0.85)
                    ax2.fill_between(t_cD, cD, alpha=0.08, color=wcol)
                    ax2.set_title(f"{wname} · cD{actual_lvl} (detail lvl {actual_lvl})")
                    ax2.grid(True)
                    if col_idx == 0:
                        ax2.set_ylabel(f"cD{actual_lvl}", fontsize=8, color=wcol)
                    if row == n_rows - 1:
                        ax2.set_xlabel("Czas [s]")

            except Exception as e:
                st.error(f"Błąd dla falki {wname}: {e}")

    st.pyplot(fig)

    # ── Rekonstrukcja ─────────────────────────────────────────────────────────
    st.markdown("#### 🔄 Rekonstrukcja sygnału")
   

    with plt.rc_context(PLT):
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 3.5))
        fig2.patch.set_facecolor("#0a0e14")

        for i, (wname, wcol) in enumerate(zip(wav_list, WAV_COLORS)):
            try:
                coeffs = pywt.wavedec(x_c, wname, level=level)
                x_rec  = pywt.waverec(coeffs, wname)
                # Wyrównaj długość
                min_len = min(len(x_c), len(x_rec))
                err = np.max(np.abs(x_c[:min_len] - x_rec[:min_len]))

                axes2[i].plot(t_c[:500], x_c[:500], color="#e6edf3",
                              lw=1.0, alpha=0.5, label="oryginał")
                axes2[i].plot(t_c[:500], x_rec[:500], color=wcol,
                              lw=1.2, ls="--", label=f"rekonstrukcja")
                axes2[i].set_title(f"{wname} · błąd maks = {err:.2e}")
                axes2[i].set_xlabel("Czas [s]"); axes2[i].grid(True)
                axes2[i].legend()
            except Exception as e:
                axes2[i].text(0.5, 0.5, f"Błąd:\n{e}",
                              ha='center', va='center', transform=axes2[i].transAxes)

        plt.tight_layout()
    st.pyplot(fig2)

    with st.expander("📖 Jak czytać dekompozycję DWT?"):
        st.markdown(f"""
        Przy fs={fs_c} Hz i {level} poziomach dekompozycji, każdy poziom cD obejmuje pasmo:

        | Poziom | Składowa | Pasmo częstotliwości (przybliżenie) |
        |---|---|---|
        """ + "\n".join([
            f"| {level} | cD{level} (detail) | {fs_c/2**(level):.0f} – {fs_c/2**(level-1) if level > 1 else fs_c/2:.0f} Hz |"
        ] + [
            f"| {level-i} | cD{level-i} | {fs_c/2**(level-i+1):.0f} – {fs_c/2**(level-i):.0f} Hz |"
            for i in range(1, level)
        ] + [
            f"| — | cA{level} | 0 – {fs_c/2**(level):.0f} Hz |"
        ]) + """

        **Dlaczego chirp jest idealnym testem?**
        Częstotliwość chirp rośnie liniowo → różne cD "zapalają się" w różnym momencie czasowym.
        Widać to jako przesunięte w czasie aktywacje na kolejnych poziomach.
        """)