import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import warnings

# wylaczenie ostrzezen bo zasmiecaja konsole
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    # 1
    nazwa_we = input("podaj nazwe pliku do wczytania")
    
    img_bgr = cv2.imread(nazwa_we)
    
    # zabezpieczenie jakbys wpisal zla nazwe
    if img_bgr is None:
        print(f"BRAK PLIKU '{nazwa_we}'eeeee")
        img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(300): img_bgr[:, i] = (i%255, 150, 255 - i%255)
        
    # konwersja do RGB zeby kolory w matplotlibie sie nie posypaly (nie byl niebieski)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # pokazujemy wczytany obraz
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(img_rgb)
    ax1.set_title(f'Zad 1: Wczytany obraz ({nazwa_we})')
    ax1.axis('off')
    print("ZAMKNIJ OKNO")
    plt.show()
    
    # po zamknieciu okna prosimy o zapis
    nazwa_wy = input("podaj nazwe pliku do zapisu: ")
    cv2.imwrite(nazwa_wy, img_bgr)
    print(f"zapisano jako {nazwa_wy}!")

    # 2
    print("histogram... (zamknij okno)")
    
    # wiersz, 2 kolumny (lewa: fotka, prawa: histogram)
    fig2, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Lewy panel - wyswietlamy oryginalny obraz
    ax_img.imshow(img_rgb)
    ax_img.set_title('Zad 2: Wczytany obraz')
    ax_img.axis('off')

    # Prawy panel - prawdziwy histogram (slupki)
    kolory = ('red', 'green', 'blue')
    for i, col in enumerate(kolory):
        # liczymy histogram dla kazdego kanalu rgb osobno
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        
        # uzywamy bar zamiast plot. 
        # hist.flatten() jest super wazne, bo opencv zwraca tablice 2D, a bar chce 1D
        ax_hist.bar(np.arange(256), hist.flatten(), color=col, alpha=0.5, width=1)
        
    ax_hist.set_title('Zad 2: Histogram RGB')
    ax_hist.set_xlim([0, 256])
    ax_hist.set_xlabel('Wartosc piksela (0-255)')
    ax_hist.set_ylabel('Liczba pikseli')
    
    plt.tight_layout()
    plt.show()

    # 3
    print("panel sol i pieprz... (zamknij okno )")
    
    fig3, ax_sp = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    
    sp_plot = ax_sp.imshow(img_rgb)
    ax_sp.set_title('Zad 3: Szum Sol i Pieprz')
    ax_sp.axis('off')

    def dodaj_sp(obraz, prob):
        if prob == 0: return obraz
        losowe = np.random.rand(*obraz.shape[:2])
        zaszumiony = obraz.copy()
        zaszumiony[losowe < prob/2] = [0, 0, 0]          # pieprz
        zaszumiony[losowe > 1 - prob/2] = [255, 255, 255] # sol
        return zaszumiony

    ax_slider_sp = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_sp = Slider(ax_slider_sp, 'Natezenie', 0.0, 0.5, valinit=0.0)

    def update_sp(val):
        nowy_obraz = dodaj_sp(img_rgb, slider_sp.val)
        sp_plot.set_data(nowy_obraz)
        fig3.canvas.draw_idle()

    slider_sp.on_changed(update_sp)
    plt.show()

    # 4
    print("panel plynnego obrotu... (zamknij okno)")
    
    # robimy kopie oryginalu. TO WAZNE: bedziemy zawsze obracac ten bazowy obraz, 
    # a nie ten juz obrocony, zeby nie psuc jakosci pikseli przy kazdym ruchu suwaka
    obraz_baza = img_rgb.copy()
    h, w = obraz_baza.shape[:2]
    srodek = (w // 2, h // 2) # wyznaczamy srodek obrazu
    
    fig4, ax_rot = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)
    
    rot_plot = ax_rot.imshow(obraz_baza)
    ax_rot.set_title('Zad 4: Obrot o dowolny kat')
    ax_rot.axis('off')

    # os na suwak (zamiast przycisku)
    ax_slider_kat = plt.axes([0.15, 0.08, 0.7, 0.03])
    # suwak od 0 do 360 stopni, skok co 1 stopien
    slider_kat = Slider(ax_slider_kat, 'Kąt', 0.0, 360.0, valinit=0.0, valstep=1.0)

    def update_obrot(val):
        kat = slider_kat.val
        
        # 1. generujemy macierz obrotu dla danego kata wokol srodka obrazka (skala = 1.0)
        M = cv2.getRotationMatrix2D(srodek, -kat, 1.0) 
        
        # 2. przeliczamy piksele (warpAffine). borderValue to kolor tla, ktore sie odslania (tutaj biale)
        obrocony = cv2.warpAffine(obraz_baza, M, (w, h), borderValue=(255, 255, 255))
        
        rot_plot.set_data(obrocony)
        fig4.canvas.draw_idle()

    slider_kat.on_changed(update_obrot)
    plt.show()



    # 5.1
    print("panel szumow... (zamknij zeby zakonczyc)")
    
    fig5, (ax_ric, ax_poi, ax_ray) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.35)

    def rician(v, s, size=10000):
        x = np.random.normal(v, s, size)
        y = np.random.normal(0, s, size)
        return np.sqrt(x**2 + y**2)

    def rysuj_szumy():
        ax_ric.clear(); ax_poi.clear(); ax_ray.clear()
        
        # 1. Rician
        szum_r = rician(s_ric_v.val, s_ric_s.val)
        ax_ric.hist(szum_r, bins=50, color='purple', density=True)
        ax_ric.set_title('Rician')
        ax_ric.set_xlim(0, 15)  # Zamrozenie osi X
        ax_ric.set_ylim(0, 1.2) # Zamrozenie osi Y
        
        # 2. Poisson
        szum_p = np.random.poisson(s_poi_lam.val, 10000)
        ax_poi.hist(szum_p, bins=50, color='orange', density=True)
        ax_poi.set_title('Poisson')
        ax_poi.set_xlim(0, 35)  # Poisson z Lambda=20 doleci max do ok. 35
        ax_poi.set_ylim(0, 0.4) # Max wysokosc dla Lambdy 1 wynosi ok 0.36
        
        # 3. Rayleigh
        szum_ray = np.random.rayleigh(s_ray_sc.val, 10000)
        ax_ray.hist(szum_ray, bins=50, color='teal', density=True)
        ax_ray.set_title('Rayleigh')
        ax_ray.set_xlim(0, 20)  # Dla najwiekszej skali rozciagnie sie do ok 20
        ax_ray.set_ylim(0, 1.5) # Zablokowanie wysokosci

    ax_sv = plt.axes([0.1, 0.20, 0.2, 0.02]); s_ric_v = Slider(ax_sv, 'v (sila sygnalu)', 0.0, 5.0, valinit=2.0)
    ax_ss = plt.axes([0.1, 0.15, 0.2, 0.02]); s_ric_s = Slider(ax_ss, 's (rozrzut)', 0.1, 3.0, valinit=1.0)
    
    ax_pl = plt.axes([0.4, 0.20, 0.2, 0.02]); s_poi_lam = Slider(ax_pl, 'Lambda', 1.0, 20.0, valinit=5.0)
    ax_rs = plt.axes([0.7, 0.20, 0.2, 0.02]); s_ray_sc = Slider(ax_rs, 'Scale', 0.1, 5.0, valinit=1.0)

    def update_szumy(val):
        rysuj_szumy()
        fig5.canvas.draw_idle()

    for s in [s_ric_v, s_ric_s, s_poi_lam, s_ray_sc]: s.on_changed(update_szumy)


    # 5.2
    print("panel przebiegow czasowych... (zamknij zeby zakonczyc)")

    fig6, (ax_t_ric, ax_t_poi, ax_t_ray) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25) # miejsce na guzik

    # przycisk do losowania [lewo, dol, szerokosc, wysokosc]
    ax_btn_los = plt.axes([0.4, 0.05, 0.2, 0.08])
    btn_los = Button(ax_btn_los, 'Nowe parametry')

    def rysuj_przebiegi(event):
        ax_t_ric.clear(); ax_t_poi.clear(); ax_t_ray.clear()
        
        los_v = np.random.uniform(0.0, 5.0)
        los_s = np.random.uniform(0.1, 3.0)
        los_lam = np.random.randint(1, 20)
        los_sc = np.random.uniform(0.1, 5.0)
        
        ilosc_probek = 200 
        
        # 1. Rician
        szum_r = rician(los_v, los_s, ilosc_probek)
        ax_t_ric.plot(szum_r, color='purple', linewidth=1)
        ax_t_ric.set_title(f'Rician (v={los_v:.1f}, s={los_s:.1f})')
        ax_t_ric.set_xlabel('probka')
        ax_t_ric.set_ylabel('amplituda')
        
        # 2. Poisson
        szum_p = np.random.poisson(los_lam, ilosc_probek)
        ax_t_poi.plot(szum_p, color='orange', linewidth=1)
        ax_t_poi.set_title(f'Poisson (lam={los_lam})')
        ax_t_poi.set_xlabel('probka')
        
        # 3. Rayleigh
        szum_ray = np.random.rayleigh(los_sc, ilosc_probek)
        ax_t_ray.plot(szum_ray, color='teal', linewidth=1)
        ax_t_ray.set_title(f'Rayleigh (scale={los_sc:.1f})')
        ax_t_ray.set_xlabel('probka')
        
        fig6.canvas.draw_idle()

    rysuj_przebiegi(None)
    btn_los.on_clicked(rysuj_przebiegi)
    
    plt.show()


    rysuj_szumy()
    plt.show()
