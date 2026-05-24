import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

# custom
st.set_page_config(
    page_title="Lista 5 · Widma sygnałów",
    page_icon="〜",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}

/* Ciemniejszy, techniczny wygląd */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0d1117;
    padding: 6px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    background: transparent;
    border-radius: 6px;
    padding: 8px 14px;
}
.stTabs [aria-selected="true"] {
    background: #1f2937 !important;
    color: #58a6ff !important;
}

.info-box {
    background: #0d1117;
    border-left: 3px solid #58a6ff;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 16px;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.6;
}
.formula {
    background: #161b22;
    border: 1px solid #30363d;
    padding: 8px 14px;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #79c0ff;
    margin: 8px 0;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

st.title("Lista 5 · Widma sygnałów")
st.caption("Przetwarzanie strumieni danych · FFT i analiza widmowa")

#

FIGSIZE = (11, 4.5)
PLT_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "text.color": "#c9d1d9",
    "axes.titlecolor": "#58a6ff",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
}

def styled_fig(nrows=1, ncols=2, figsize=FIGSIZE):
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    return fig, axes

def compute_fft(x, fs):
    """Zwraca jednostronne widmo: (freqs, amplitudy)."""
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1/fs)
    # Tylko dodatnie częstotliwości
    half = N // 2
    freqs = freqs[:half]
    amp = (2 / N) * np.abs(X[:half])
    return freqs, amp

def plot_signal(ax, t, x, title="Sygnał", color="#58a6ff"):
    ax.plot(t, x, color=color, lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")
    ax.grid(True)

def plot_spectrum(ax, freqs, amp, title="Widmo amplitudowe", color="#f78166"):
    ax.stem(freqs, amp, linefmt=color, markerfmt=f"o", basefmt=" ")
    ax.set_title(title)
    ax.set_xlabel("Częstotliwość [Hz]")
    ax.set_ylabel("|X(f)|")
    ax.grid(True)

# 
tabs = st.tabs([
    "Zad 1 · Widmo liniowe",
    "Zad 2 · Widmo w dB",
    "Zad 3 · Wiele sygnałów",
    "Zad 4 · Widmo fazowe",
    "Zad 5 · Obwiednia chirp",
])

# 1
with tabs[0]:

    c1, c2, c3, c4 = st.columns(4)
    f0   = c1.slider("Częstotliwość f₀ [Hz]", 1, 100, 10, key="z1f")
    A    = c2.slider("Amplituda A", 0.1, 5.0, 1.0, 0.1, key="z1a")
    fs   = c3.slider("Próbkowanie fs [Hz]", 200, 5000, 1000, 50, key="z1fs")
    dur  = c4.slider("Czas trwania [s]", 0.1, 5.0, 1.0, 0.1, key="z1d")

    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    x = A * np.sin(2 * np.pi * f0 * t)
    freqs, amp = compute_fft(x, fs)

    with plt.rc_context(PLT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
        fig.patch.set_facecolor("#0d1117")
        plot_signal(ax1, t[:int(fs*min(dur,0.1))], x[:int(fs*min(dur,0.1))],
                    title=f"Sinus f₀={f0} Hz, A={A}")
        ax2.stem(freqs, amp, linefmt="#f78166", markerfmt="o", basefmt=" ")
        ax2.set_title("Widmo amplitudowe |X(f)|")
        ax2.set_xlabel("Częstotliwość [Hz]"); ax2.set_ylabel("|X(f)|")
        ax2.set_xlim(0, min(fs/2, f0*5+50)); ax2.grid(True)
        plt.tight_layout()
    st.pyplot(fig)

# 2
with tabs[1]:

    c1, c2, c3, c4 = st.columns(4)
    f0   = c1.slider("Częstotliwość f₀ [Hz]", 1, 100, 10, key="z2f")
    A    = c2.slider("Amplituda A", 0.1, 5.0, 1.0, 0.1, key="z2a")
    fs   = c3.slider("Próbkowanie fs [Hz]", 200, 5000, 1000, 50, key="z2fs")
    noise = c4.slider("Szum (SNR dB)", 0, 60, 40, key="z2n")

    dur = 1.0
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    noise_amp = A * 10 ** (-noise / 20)
    x = A * np.sin(2 * np.pi * f0 * t) + noise_amp * np.random.randn(len(t))

    freqs, amp = compute_fft(x, fs)
    amp_db = 20 * np.log10(amp + 1e-12)

    with plt.rc_context(PLT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
        fig.patch.set_facecolor("#0d1117")
        ax1.plot(freqs, amp, color="#f78166", lw=1.2)
        ax1.set_title("Widmo liniowe |X(f)|")
        ax1.set_xlabel("Częstotliwość [Hz]"); ax1.set_ylabel("|X(f)|"); ax1.grid(True)

        ax2.plot(freqs, amp_db, color="#3fb950", lw=1.2)
        ax2.set_title("Widmo w skali dB  →  20·log₁₀|X(f)|")
        ax2.set_xlabel("Częstotliwość [Hz]"); ax2.set_ylabel("Amplituda [dB]"); ax2.grid(True)
        ax2.set_xlim(0, fs / 2)
        plt.tight_layout()
    st.pyplot(fig)

# 3
with tabs[2]:

    c1, c2, c3 = st.columns(3)
    f0  = c1.slider("Częstotliwość f₀ [Hz]", 1, 100, 5, key="z3f")
    fs  = c2.slider("Próbkowanie fs [Hz]", 200, 5000, 1000, 50, key="z3fs")
    dur = c3.slider("Czas [s]", 0.5, 5.0, 2.0, 0.5, key="z3d")

    t = np.linspace(0, dur, int(fs * dur), endpoint=False)

    signals = {
        "Sinus":              np.sin(2 * np.pi * f0 * t),
        "Prostokątny":        signal.square(2 * np.pi * f0 * t),
        "Piłokształtny":      signal.sawtooth(2 * np.pi * f0 * t),
        "Chirp (świergot)":   signal.chirp(t, f0=f0, f1=f0*5, t1=dur, method='linear'),
        "Sin+Cos (superpoz)": np.sin(2*np.pi*f0*t) + 0.5*np.cos(2*np.pi*f0*2*t),
        "Impuls jedn.":       signal.unit_impulse(len(t), idx=len(t)//4),
    }

    COLORS = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657", "#ff7b72"]

    with plt.rc_context(PLT_STYLE):
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor("#0d1117")
        gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.85, wspace=0.35)

        for i, (name, sig) in enumerate(signals.items()):
            col = COLORS[i]
            freqs, amp = compute_fft(sig, fs)

            ax_t = fig.add_subplot(gs[i, 0])
            ax_t.plot(t[:int(fs*min(dur, 0.5))], sig[:int(fs*min(dur, 0.5))],
                      color=col, lw=1.0)
            ax_t.set_title(f"{name} — sygnał", fontsize=9)
            ax_t.set_xlabel("t [s]", fontsize=8); ax_t.grid(True)

            ax_f = fig.add_subplot(gs[i, 1])
            ax_f.plot(freqs, amp, color=col, lw=1.0)
            ax_f.set_title(f"{name} — |X(f)|", fontsize=9)
            ax_f.set_xlabel("f [Hz]", fontsize=8)
            ax_f.set_xlim(0, min(fs/2, f0*20+50)); ax_f.grid(True)

    st.pyplot(fig)

   
# 4

with tabs[3]:
   

    c1, c2, c3, c4 = st.columns(4)
    f0     = c1.slider("Częstotliwość f₀ [Hz]", 1, 100, 10, key="z4f")
    phase  = c2.slider("Faza sygnału φ [°]", -180, 180, 0, key="z4p")
    A      = c3.slider("Amplituda A", 0.1, 5.0, 1.0, 0.1, key="z4a")
    fs     = c4.slider("Próbkowanie fs [Hz]", 200, 5000, 1000, 50, key="z4fs")

    dur = 1.0
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    x = A * np.sin(2 * np.pi * f0 * t + np.deg2rad(phase))

    N = len(x)
    X = np.fft.fft(x)
    half = N // 2
    freqs = np.fft.fftfreq(N, d=1/fs)[:half]
    amp   = (2 / N) * np.abs(X[:half])
    pha   = np.angle(X[:half], deg=True)

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor("#0d1117")

        plot_signal(axes[0], t[:200], x[:200], title=f"Sygnał (φ={phase}°)")

        axes[1].stem(freqs, amp, linefmt="#f78166", markerfmt="o", basefmt=" ")
        axes[1].set_title("Widmo amplitudowe |X(f)|")
        axes[1].set_xlabel("f [Hz]"); axes[1].set_ylabel("|X(f)|")
        axes[1].set_xlim(0, f0*4+20); axes[1].grid(True)

        axes[2].stem(freqs, pha, linefmt="#d2a8ff", markerfmt="o", basefmt=" ")
        axes[2].set_title("Widmo fazowe ∠X(f) [°]")
        axes[2].set_xlabel("f [Hz]"); axes[2].set_ylabel("Faza [°]")
        axes[2].set_xlim(0, f0*4+20); axes[2].set_ylim(-200, 200)
        axes[2].axhline(0, color="#30363d", lw=0.8)
        axes[2].grid(True)

        plt.tight_layout()
    st.pyplot(fig)


# 5
with tabs[4]:
    

    c1, c2, c3, c4, c5 = st.columns(5)
    f_start = c1.slider("f_start [Hz]", 1, 200, 10, key="z5fs")
    f_end   = c2.slider("f_end [Hz]", 10, 1000, 200, key="z5fe")
    dur     = c3.slider("Czas [s]", 0.5, 5.0, 2.0, 0.5, key="z5d")
    fs      = c4.slider("Próbkowanie [Hz]", 500, 8000, 4000, 100, key="z5sr")

    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    x = signal.chirp(t, f0=f_start, f1=f_end, t1=dur)

    # hilbert -> obwiednia
    analytic = signal.hilbert(x)
    envelope = np.abs(analytic)
    inst_freq = np.diff(np.unwrap(np.angle(analytic))) / (2 * np.pi / fs)

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.patch.set_facecolor("#0d1117")

        # sygnał + obwiednia
        axes[0].plot(t, x, color="#58a6ff", lw=0.7, alpha=0.7, label="chirp")
        axes[0].plot(t, envelope,  color="#f78166", lw=2.0, label="obwiednia (+)")
        axes[0].plot(t, -envelope, color="#f78166", lw=2.0, label="obwiednia (−)")
        axes[0].set_title("Sygnał chirp + obwiednia")
        axes[0].set_xlabel("Czas [s]"); axes[0].set_ylabel("Amplituda")
        axes[0].legend(fontsize=8); axes[0].grid(True)

        # spektrogram
        f_spec, t_spec, Sxx = signal.spectrogram(x, fs, nperseg=256)
        axes[1].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-12),
                           shading='gouraud', cmap='inferno')
        axes[1].set_title("Spektrogram (czas × częstotliwość)")
        axes[1].set_xlabel("Czas [s]"); axes[1].set_ylabel("Częstotliwość [Hz]")
        axes[1].set_ylim(0, min(fs/2, f_end * 1.5))

        # chwilowa częstotliwość
        axes[2].plot(t[:-1], inst_freq, color="#3fb950", lw=1.5)
        axes[2].set_title("Chwilowa częstotliwość")
        axes[2].set_xlabel("Czas [s]"); axes[2].set_ylabel("f(t) [Hz]")
        axes[2].set_ylim(0, f_end * 1.2); axes[2].grid(True)

        plt.tight_layout()
    st.pyplot(fig)