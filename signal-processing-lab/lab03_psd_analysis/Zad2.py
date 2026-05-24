import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Slider, RadioButtons
import warnings

warnings.filterwarnings('ignore')

# Wektor czasu jest teraz generowany dynamicznie wewnątrz funkcji w zależności od fs
def generuj_sygnal(typ, freq, fs_current):
    # Generujemy czas t dla 1 sekundy
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
    
    # Liczymy tylko dla dodatnich częstotliwości (od 0 do fs/2)
    K = N // 2 + 1 
    
    Sxx = np.zeros(K)
    f = np.zeros(K)
    
    for k in range(K):
        kat = 2 * np.pi * k * n / N

        suma_re = np.sum(rxx * np.cos(kat))
        suma_im = np.sum(rxx * (-np.sin(kat)))
        
        # Moduł z liczby zespolonej to pierwiastek z sumy kwadrat
        Sxx[k] = np.sqrt(suma_re**2 + suma_im**2)
        f[k] = k * (fs / N)
        
    return f, Sxx


if __name__ == "__main__":

    print("\n\n--- ZAD 2: WGM z definicji (Ręczna suma funkcji) ---")
    print("Odpalam interfejs... (zamknij okno aby przejsc do Zad 3)")

    fig2, ax_def2 = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # 1. Wybór sygnału
    ax_radio2 = plt.axes([0.02, 0.55, 0.15, 0.35])
    radio2 = RadioButtons(ax_radio2, ('sinus', 'prostokat', 'pila', 'swiergot', 'mix', 'impuls'))
    ax_radio2.set_title("Typ Sygnału")
    
    # 2. Skala Y (Log / Lin)
    ax_scale = plt.axes([0.02, 0.35, 0.15, 0.15])
    radio_scale = RadioButtons(ax_scale, ('Log', 'Lin'))
    ax_scale.set_title("Skala Y")
    
    # 3. Suwaki
    ax_freq2 = plt.axes([0.3, 0.10, 0.6, 0.03])
    # Domyślnie startujemy z 40Hz zeby lepiej było widać wpływ małego fs
    s_freq2 = Slider(ax_freq2, 'Freq [Hz]', 1.0, 200.0, valinit=40.0)
    
    ax_fs = plt.axes([0.3, 0.05, 0.6, 0.03])
    s_fs = Slider(ax_fs, 'Próbkowanie (fs)', 100.0, 2000.0, valinit=1000.0, valstep=10.0)

    def update_zad2(val):
        fs_val = s_fs.val
        f_sig = s_freq2.val
        use_log = radio_scale.value_selected == 'Log'
        
        # Generowanie sygnału z dynamicznym fs
        x = generuj_sygnal(radio2.value_selected, f_sig, fs_val)
        
        # Wywołanie naszej ręcznej funkcji
        f_def, Sxx = psd_z_definicji_suma(x, fs_val)
        
        ax_def2.clear()
        
        if use_log:
            ax_def2.semilogy(f_def, Sxx, color='red')
            ax_def2.set_ylim([1e-6, 1e3]) # Zamrożona skala dla logarytmu
        else:
            ax_def2.plot(f_def, Sxx, color='red')
            ax_def2.set_ylim([0, np.max(Sxx) * 1.2 + 1e-9]) # Zapas z góry dla skali liniowej

        ax_def2.set_title(f'Zad 2: WGM z definicji (Suma Eulera) - fs={int(fs_val)}Hz')
        ax_def2.set_xlim([0, fs_val/2]) # Zabezpieczenie osi X do f_Nyquista
        ax_def2.set_ylabel('PSD')
        ax_def2.set_xlabel('Częstotliwość [Hz]')
        
        fig2.canvas.draw_idle()

    # Podpinanie eventów
    radio2.on_clicked(update_zad2)
    radio_scale.on_clicked(update_zad2)
    s_freq2.on_changed(update_zad2)
    s_fs.on_changed(update_zad2)
    
    update_zad2(None)
    plt.show()