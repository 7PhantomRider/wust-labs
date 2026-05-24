import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Slider, RadioButtons
import warnings

warnings.filterwarnings('ignore')

# Funkcja generująca sygnał - czas t jest teraz zależny od fs
def generuj_sygnal(typ, freq, fs_current):
    # Generujemy wektor czasu dla 1 sekundy przy aktualnym fs
    t_current = np.linspace(0, 1, int(fs_current), endpoint=False)
    
    if typ == 'sinus': 
        x = np.sin(2 * np.pi * freq * t_current)
    elif typ == 'prostokat': 
        x = signal.square(2 * np.pi * freq * t_current)
    elif typ == 'pila': 
        x = signal.sawtooth(2 * np.pi * freq * t_current)
    elif typ == 'swiergot': 
        x = signal.chirp(t_current, f0=1, f1=freq*2, t1=1, method='linear')
    elif typ == 'mix': 
        x = np.sin(2 * np.pi * freq * t_current) + 0.5 * np.cos(2 * np.pi * (freq * 2.5) * t_current)
    elif typ == 'impuls': 
        x = signal.unit_impulse(int(fs_current), 'mid')
    return t_current, x

if __name__ == "__main__":
    print("\n--- ZAD 1: Analiza wpływu próbkowania (fs) ---")
    
    fig1, (ax_per, ax_welch) = plt.subplots(1, 2, figsize=(13, 6))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # 1. Wybór Sygnału
    ax_radio = plt.axes([0.02, 0.60, 0.15, 0.30])
    radio_sig = RadioButtons(ax_radio, ('sinus', 'prostokat', 'pila', 'swiergot', 'mix', 'impuls'))
    ax_radio.set_title("Typ Sygnału")

    # 2. Skala Y (Log / Lin)
    ax_scale = plt.axes([0.02, 0.35, 0.15, 0.15])
    radio_scale = RadioButtons(ax_scale, ('Log', 'Lin'))
    ax_scale.set_title("Skala Y")
    
    # 3. Suwaki
    ax_freq = plt.axes([0.3, 0.10, 0.6, 0.03])
    s_freq = Slider(ax_freq, 'Freq Sygnału [Hz]', 1.0, 200.0, valinit=40.0) 
    
    ax_fs = plt.axes([0.3, 0.05, 0.6, 0.03])
    s_fs = Slider(ax_fs, 'Próbkowanie (fs) [Hz]', 100.0, 2000.0, valinit=1000.0, valstep=10.0)

    def update(val):
        fs_val = s_fs.val
        f_sig = s_freq.val
        use_log = radio_scale.value_selected == 'Log'
        
        # Generujemy sygnał dla aktualnych ustawień
        t_curr, x = generuj_sygnal(radio_sig.value_selected, f_sig, fs_val)
        
        # Obliczenia PSD
        f_p, p_p = signal.periodogram(x, fs_val)
        
        # Dla Welcha nperseg nie może być większy niż liczba próbek
        n_seg = min(256, len(x))
        f_w, p_w = signal.welch(x, fs_val, nperseg=n_seg)
        
        # Wykresy
        for ax, f, p, title, color in [(ax_per, f_p, p_p, 'Periodogram', 'blue'), 
                                       (ax_welch, f_w, p_w, 'Welch', 'green')]:
            ax.clear()
            if use_log:
                ax.semilogy(f, p, color=color)
                ax.set_ylim([1e-10, 1e2]) # Stała skala dla LOG
            else:
                ax.plot(f, p, color=color)
                ax.set_ylim([0, np.max(p_p) * 1.2 + 1e-9]) # Dopasowanie dla LIN
            
            ax.set_title(f'{title} (fs={int(fs_val)}Hz)')
            ax.set_xlim([0, fs_val/2]) # Oś X do częstotliwości Nyquista
            ax.set_xlabel('f [Hz]')
        
        ax_per.set_ylabel('PSD')
        fig1.canvas.draw_idle()

    radio_sig.on_clicked(update)
    radio_scale.on_clicked(update)
    s_freq.on_changed(update)
    s_fs.on_changed(update)
    
    update(None)
    plt.show()