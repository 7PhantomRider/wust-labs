import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Slider, RadioButtons
import warnings

warnings.filterwarnings('ignore')

def generuj_sygnal(typ, freq, fs_current):
    t_current = np.linspace(0, 1, int(fs_current), endpoint=False)
    
    if typ == 'sinus': x = np.sin(2 * np.pi * freq * t_current)
    elif typ == 'prostokat': x = signal.square(2 * np.pi * freq * t_current)
    elif typ == 'pila': x = signal.sawtooth(2 * np.pi * freq * t_current)
    elif typ == 'swiergot': x = signal.chirp(t_current, f0=1, f1=freq*2, t1=1, method='linear')
    elif typ == 'mix': x = np.sin(2 * np.pi * freq * t_current) + 0.5 * np.cos(2 * np.pi * (freq * 2.5) * t_current)
    elif typ == 'impuls': x = signal.unit_impulse(int(fs_current), 'mid')
    
    return x

def psd_z_definicji_suma(x, fs):
    rxx = np.correlate(x, x, mode='full') / len(x)
    N = len(rxx)
    n = np.arange(N)
    
    K = N // 2 + 1 
    Sxx = np.zeros(K)
    f = np.zeros(K)
    
    for k in range(K):
        kat = 2 * np.pi * k * n / N
        suma_re = np.sum(rxx * np.cos(kat))
        suma_im = np.sum(rxx * (-np.sin(kat)))
        
        Sxx[k] = np.sqrt(suma_re**2 + suma_im**2)
        f[k] = k * (fs / N)
        
    return f, Sxx

if __name__ == "__main__":
    print("\n--- ZAD 3: Porównanie metod (Periodogram vs Welch vs Definicja) ---")

    # Tworzymy 3 wykresy
    fig, (ax_p, ax_w, ax_d) = plt.subplots(3, 1, figsize=(12, 8))
    plt.subplots_adjust(left=0.22, bottom=0.18, hspace=0.6)

    ax_radio = plt.axes([0.02, 0.55, 0.12, 0.3])
    radio = RadioButtons(ax_radio, ('sinus', 'prostokat', 'pila', 'swiergot', 'mix', 'impuls'))
    ax_radio.set_title("Typ Sygnału") 
    
    ax_scale = plt.axes([0.02, 0.35, 0.12, 0.15])
    radio_scale = RadioButtons(ax_scale, ('Log', 'Lin'))
    ax_scale.set_title("Skala Y")       
    
    ax_freq = plt.axes([0.25, 0.10, 0.6, 0.02])
    s_freq = Slider(ax_freq, 'Freq [Hz]', 1.0, 200.0, valinit=40.0)
    
    ax_fs = plt.axes([0.25, 0.05, 0.6, 0.02])
    s_fs = Slider(ax_fs, 'Próbkowanie (fs)', 100.0, 2000.0, valinit=1000.0, valstep=10.0)

    def update(val):
        fs_val = s_fs.val
        f_sig = s_freq.val
        use_log = radio_scale.value_selected == 'Log'
        
        # 1. Generowanie czystego sygnału
        x = generuj_sygnal(radio.value_selected, f_sig, fs_val)
        
        # Dodajemy bardzo delikatny szum w locie, żeby Welch miał co uśredniać
        x_zaszumione = x + 0.3 * np.random.randn(len(x))
        
        # 2. Obliczenia (Wszystkie 3 metody)
        f_per, Pxx_per = signal.periodogram(x_zaszumione, fs_val)
        
        n_seg = min(256, len(x_zaszumione))
        f_welch, Pxx_welch = signal.welch(x_zaszumione, fs_val, nperseg=n_seg)
        
        f_def, Sxx_def = psd_z_definicji_suma(x_zaszumione, fs_val)
        
        plots = [(ax_p, f_per, Pxx_per, '1. Periodogram (SciPy)', 'blue'),
                 (ax_w, f_welch, Pxx_welch, '2. Metoda Welcha (SciPy)', 'green'),
                 (ax_d, f_def, Sxx_def, '3. Z definicji (Ręczna suma)', 'red')]
        
        # 3. Rysowanie widma
        for ax, f, p, title, color in plots:
            ax.clear()
            
            if use_log:
                ax.semilogy(f, p, color=color)
                # Twarda podłoga dla logarytmu, żeby wykresy nie skakały
                ax.set_ylim([1e-6, 1e4]) 
            else:
                ax.plot(f, p, color=color)
                # Skala LINIOWA: Każdy wykres oblicza swoje własne, niezależne maksimum!
                lokalne_max = np.max(p)
                ax.set_ylim([0, lokalne_max * 1.2 + 1e-9])

            ax.set_title(f'{title} (fs={int(fs_val)}Hz)')
            ax.set_xlim([0, fs_val/2])
            ax.set_ylabel('PSD')
            
        ax_d.set_xlabel('Częstotliwość [Hz]')
        fig.canvas.draw_idle()

    radio.on_clicked(update)
    radio_scale.on_clicked(update)
    s_freq.on_changed(update)
    s_fs.on_changed(update)
    
    # Inicjalne uruchomienie
    update(None)
    plt.show()