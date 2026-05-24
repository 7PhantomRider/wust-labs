import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Slider, RadioButtons, Button

if __name__ == "__main__":

    # wektor czasu wspolny
    t = np.linspace(0, 1, 1000, endpoint=False)




    # 1
    s_sin = np.sin(2 * np.pi * 5 * t)
    s_sqr = signal.square(2 * np.pi * 5 * t)
    s_saw = signal.sawtooth(2 * np.pi * 5 * t)
    s_chirp = signal.chirp(t, f0=1, f1=10, t1=1, method='linear')
    s_mix = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 8 * t)
    s_imp = signal.unit_impulse(1000, 'mid')

    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    axs[0, 0].plot(t, s_sin); axs[0, 0].set_title('sinus')
    axs[0, 1].plot(t, s_sqr); axs[0, 1].set_title('prostokatny')
    axs[1, 0].plot(t, s_saw); axs[1, 0].set_title('piloksztaltny')
    axs[1, 1].plot(t, s_chirp); axs[1, 1].set_title('swiergot')
    axs[2, 0].plot(t, s_mix); axs[2, 0].set_title('superpozycja')
    axs[2, 1].plot(t, s_imp); axs[2, 1].set_title('impuls')
    plt.tight_layout()
    plt.show()


    # osobno
    sygnaly = [
        ('sinus', s_sin, 'blue'),
        ('prostokatny', s_sqr, 'orange'),
        ('piloksztaltny', s_saw, 'green'),
        ('swiergot', s_chirp, 'red'),
        ('superpozycja', s_mix, 'purple'),
        ('impuls', s_imp, 'brown')
    ]

    for nazwa, dane, kolor in sygnaly:
        plt.figure(figsize=(8, 4))
        plt.plot(t, dane, color=kolor)
        plt.title(f'{nazwa}')
        plt.xlabel('czas [s]')
        plt.ylabel('amplituda')
        plt.show()

    # 2 i 3
    print("\n\n\n2 i 3")
    df = pd.DataFrame({'czas': t, 'sygnal': s_sin})
    
    # prosimy w konsoli o nazwe pliku do zapisu
    nazwa_zapis = input("podaj nazwe pliku do zapisu (np. test.csv): ")
    df.to_csv(nazwa_zapis, index=False)
    print(f"zapisano dane do pliku: {nazwa_zapis}")
    
    # prosimy w konsoli o nazwe pliku do odczytu
    nazwa_odczyt = input("podaj nazwe pliku do wczytania (ten sam lub inny): ")
    df_read = pd.read_csv(nazwa_odczyt)
    
    print("wczytane dane (pierwsze 5 wierszy):")
    print(df_read.head())


    plt.figure(figsize=(8, 4))
    plt.plot(df_read['czas'], df_read['sygnal'], color='magenta')
    plt.title(f'sygnal wczytany z pliku: {nazwa_odczyt}')
    plt.xlabel('czas [s]')
    plt.ylabel('amplituda')
    plt.show()


    # 4
    szum_rand = np.random.rand(1000)
    szum_randn = np.random.randn(1000)
    
    # 1 wiersz, 3 kolumny
    fig_zad4, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # lewy: przebieg rand
    ax1.plot(szum_rand, color='blue', alpha=0.7, linewidth=0.8)
    ax1.set_title('rand')
    ax1.set_xlabel('probka')
    ax1.set_ylabel('amplituda')
    
    # srodkowy: przebieg randn
    ax2.plot(szum_randn, color='orange', alpha=0.7, linewidth=0.8)
    ax2.set_title('randn')
    ax2.set_xlabel('probka')
    
    # prawy: wspolny histogram
    ax3.hist(szum_rand, bins=30, alpha=0.5, density=True, color='blue', label='rand')
    ax3.hist(szum_randn, bins=30, alpha=0.5, density=True, color='orange', label='randn')
    ax3.set_title('histogramy')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()





    # 5
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    plt.subplots_adjust(bottom=0.45) 

    p1 = [0.0, 1.0]
    p2 = [5.0, 2.0]
    p3 = [-4.0, 0.5]
    
    def rysuj_histogramy():
        ax3.clear()
        
        s1 = np.random.normal(loc=p1[0], scale=p1[1], size=1000)
        s2 = np.random.normal(loc=p2[0], scale=p2[1], size=1000)
        s3 = np.random.normal(loc=p3[0], scale=p3[1], size=1000)
        
        # density=True - znormalizowanie
        ax3.hist(s1, bins=30, alpha=0.5, color='blue', density=True, label=f'szum 1 (loc={p1[0]:.1f}, scale={p1[1]:.1f})')
        ax3.hist(s2, bins=30, alpha=0.5, color='orange', density=True, label=f'szum 2 (loc={p2[0]:.1f}, scale={p2[1]:.1f})')
        ax3.hist(s3, bins=30, alpha=0.5, color='green', density=True, label=f'szum 3 (loc={p3[0]:.1f}, scale={p3[1]:.1f})')
        
        ax3.set_title('5 szeregi')
        ax3.set_xlim(-15, 15) # lock osi X
        ax3.set_ylim(0, 1.0)  # lock osi Y
        ax3.legend(loc='upper right')

    rysuj_histogramy()

    # miejsce na suwaki
    ax_l1 = plt.axes([0.15, 0.30, 0.3, 0.02]); ax_s1 = plt.axes([0.6, 0.30, 0.3, 0.02])
    ax_l2 = plt.axes([0.15, 0.20, 0.3, 0.02]); ax_s2 = plt.axes([0.6, 0.20, 0.3, 0.02])
    ax_l3 = plt.axes([0.15, 0.10, 0.3, 0.02]); ax_s3 = plt.axes([0.6, 0.10, 0.3, 0.02])

    # suwaki
    sl_l1 = Slider(ax_l1, 'loc 1', -10.0, 10.0, valinit=p1[0], color='blue')
    sl_s1 = Slider(ax_s1, 'scale 1', 0.1, 5.0, valinit=p1[1], color='blue')
    
    sl_l2 = Slider(ax_l2, 'loc 2', -10.0, 10.0, valinit=p2[0], color='orange')
    sl_s2 = Slider(ax_s2, 'scale 2', 0.1, 5.0, valinit=p2[1], color='orange')
    
    sl_l3 = Slider(ax_l3, 'loc 3', -10.0, 10.0, valinit=p3[0], color='green')
    sl_s3 = Slider(ax_s3, 'scale 3', 0.1, 5.0, valinit=p3[1], color='green')

    def update_hist(val):
        p1[0] = sl_l1.val; p1[1] = sl_s1.val
        p2[0] = sl_l2.val; p2[1] = sl_s2.val
        p3[0] = sl_l3.val; p3[1] = sl_s3.val
        rysuj_histogramy()
        fig3.canvas.draw_idle()

    for suwak in [sl_l1, sl_s1, sl_l2, sl_s2, sl_l3, sl_s3]:
        suwak.on_changed(update_hist)

    plt.show()




# 6     
    fig5, (ax_line, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)

    ax_btn = plt.axes([0.4, 0.05, 0.2, 0.08])
    btn = Button(ax_btn, 'nowy')

    def update_zad6(event):
        szum_cz_1d = np.cumsum(np.random.randn(1000))
        
        # L
        ax_line.clear() # czyscimy stary wykres
        ax_line.plot(szum_cz_1d, color='brown', linewidth=1)
        ax_line.set_title('przebieg czasowy (1D)')
        ax_line.set_xlabel('numer probki')
        # P
        ax_hist.clear()
        ax_hist.hist(szum_cz_1d, bins=30, color='brown', alpha=0.7)
        ax_hist.set_title('histogram')
        
        fig5.canvas.draw_idle()
    update_zad6(None)

    # update przycisk
    btn.on_clicked(update_zad6)

    plt.show()


    # 7 red noise 2D
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)
    
    max_krokow = 5000
    x_calosc = np.cumsum(np.random.randn(max_krokow))
    y_calosc = np.cumsum(np.random.randn(max_krokow))
    
    poczatek = 100
    
    # rysujemy sladd (czerwona linia)
    linia_r, = ax4.plot(x_calosc[:poczatek], y_calosc[:poczatek], color='red', alpha=0.7)
    
    # rysujemy aktualna pozycje (czarna kropka)
    punkt, = ax4.plot([x_calosc[poczatek-1]], [y_calosc[poczatek-1]], 'ko', markersize=8)
    
    ax4.set_title('red noise 2D')
    
    # granice osi z marginem
    ax4.set_xlim(np.min(x_calosc)-5, np.max(x_calosc)+5)
    ax4.set_ylim(np.min(y_calosc)-5, np.max(y_calosc)+5)
    
    # suwak
    ax_czas = plt.axes([0.15, 0.08, 0.7, 0.03])
    # valstep=1 (nie ma polowek)
    s_czas = Slider(ax_czas, 'kroki', 10, max_krokow, valinit=poczatek, valstep=1)
    
    def update_czas(val):
        n = int(s_czas.val)
        
        # update linia
        linia_r.set_data(x_calosc[:n], y_calosc[:n])
        
        # udate punktu n
        punkt.set_data([x_calosc[n-1]], [y_calosc[n-1]])
        
        fig4.canvas.draw_idle()
        
    s_czas.on_changed(update_czas)
    
    plt.show()






    # 1 suwaki
    fig2, ax2 = plt.subplots()
    
    plt.subplots_adjust(left=0.3, bottom=0.25)
    
    # wykres startowy
    linia, = ax2.plot(t, np.sin(2 * np.pi * 5.0 * t))
    ax2.set_ylim(-3, 3)
    ax2.set_title('sygnaly')

    # widgetty
    ax_freq = plt.axes([0.3, 0.1, 0.6, 0.03])
    ax_amp  = plt.axes([0.3, 0.05, 0.6, 0.03])
    ax_radio = plt.axes([0.05, 0.4, 0.2, 0.2])

    s_freq = Slider(ax_freq, 'freq', 1.0, 20.0, valinit=5.0)
    s_amp = Slider(ax_amp, 'amp', 0.1, 3.0, valinit=1.0)
    radio = RadioButtons(ax_radio, ('sinus', 'prostokat', 'pila', 'swiergot', 'superpozycja', 'impuls'))

    # update wykresu
    def update(val):
        amp = s_amp.val
        freq = s_freq.val
        typ = radio.value_selected
        
        # liczenie nowego sygnalu od typu
        if typ == 'sinus':
            y = amp * np.sin(2 * np.pi * freq * t)
        elif typ == 'prostokat':
            y = amp * signal.square(2 * np.pi * freq * t)
        elif typ == 'pila':
            y = amp * signal.sawtooth(2 * np.pi * freq * t)
        elif typ == 'swiergot':
            y = amp * signal.chirp(t, f0=1, f1=10, t1=1, method='linear')
        elif typ == 'superpozycja':
            y = amp * (np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 8 * t))
        elif typ == 'impuls':
            y = amp * signal.unit_impulse(1000, 'mid')
            
        linia.set_ydata(y)
        fig2.canvas.draw_idle()

    # wywolanie funkcji przy poruszaniu widgetami
    s_freq.on_changed(update)
    s_amp.on_changed(update)
    radio.on_clicked(update)
    
    plt.show()