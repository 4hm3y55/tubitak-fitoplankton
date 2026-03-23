# -*- coding: utf-8 -*-
"""
=============================================================================
 KÜRESEL ISINMA → FİTOPLANKTON → BESİN ZİNCİRİ SİMÜLASYONU
 Streamlit Web Arayüzü — Python 3.10+ Uyumlu
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import requests
import io
import warnings
import sys

warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


# ══════════════════════════════════════════════════════════════════════
#  SAYFA AYARLARI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Küresel Isınma & Fitoplankton Simülasyonu",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ══════════════════════════════════════════════════════════════════════
#  SABİT TROFİK SEVİYE İSİMLERİ  (tek yerde tanımla, her yerde kullan)
# ══════════════════════════════════════════════════════════════════════

TROFIK_ISIMLER = [
    'Fitoplankton', 'Zooplankton', 'Küçük Balıklar',
    'Büyük Balıklar', 'Üst Yırtıcılar'
]
TROFIK_RENKLER = ['#27ae60', '#3498db', '#9b59b6', '#e67e22', '#c0392b']
TROFIK_IKONLAR = ['🦠', '🦐', '🐟', '🐠', '🦈']


# ══════════════════════════════════════════════════════════════════════
#  BÖLÜM 1 — NASA VERİ TOPLAYICI
# ══════════════════════════════════════════════════════════════════════

class NASAVeriToplayici:
    """NASA GISTEMP, NOAA CO₂ ve MODIS Klorofil-a verilerini toplar."""

    def __init__(self):
        self.gistemp_url = (
            "https://data.giss.nasa.gov/gistemp/tabledata_v4/"
            "GLB.Ts+dSST.csv"
        )
        self.co2_url = (
            "https://gml.noaa.gov/webdata/ccgg/trends/co2/"
            "co2_annmean_mlo.csv"
        )

    # ── yardımcı ──
    @staticmethod
    def _guvenli_float(deger):
        try:
            return float(deger)
        except (ValueError, TypeError):
            return np.nan

    # ── GISTEMP ──
    def gistemp_indir(self, durum):
        durum.update(label="NASA GISTEMP v4 indiriliyor…", state="running")
        try:
            yanit = requests.get(self.gistemp_url, timeout=15)
            yanit.raise_for_status()

            satirlar = yanit.text.split('\n')
            baslik_idx = 0
            for i, satir in enumerate(satirlar):
                if satir.startswith('Year'):
                    baslik_idx = i
                    break

            veri_metni = '\n'.join(satirlar[baslik_idx:])
            df = pd.read_csv(io.StringIO(veri_metni))

            if 'Year' in df.columns and 'J-D' in df.columns:
                df = df[['Year', 'J-D']].copy()
                df.columns = ['yil', 'sicaklik_anomalisi']
            else:
                raise ValueError("Sütunlar bulunamadı")

            df['yil'] = df['yil'].apply(self._guvenli_float)
            df['sicaklik_anomalisi'] = df['sicaklik_anomalisi'].apply(
                self._guvenli_float
            )
            df = df.dropna().reset_index(drop=True)
            df['yil'] = df['yil'].astype(int)
            df['sicaklik_anomalisi'] = df['sicaklik_anomalisi'].astype(float)

            durum.update(
                label="GISTEMP: {} yıl ({}-{})".format(
                    len(df), int(df['yil'].min()), int(df['yil'].max())
                ),
                state="complete"
            )
            return df

        except Exception:
            durum.update(
                label="GISTEMP indirilemedi — yedek veri kullanılıyor",
                state="complete"
            )
            return self._gistemp_yedek()

    def _gistemp_yedek(self):
        yillar = np.arange(1880, 2025)
        temel_egilim = np.zeros(len(yillar), dtype=float)
        for i, y in enumerate(yillar):
            if y < 1910:
                temel_egilim[i] = -0.2 + 0.003 * (y - 1880)
            elif y < 1940:
                temel_egilim[i] = -0.1 + 0.01 * (y - 1910)
            elif y < 1970:
                temel_egilim[i] = 0.1 + 0.002 * (y - 1940)
            elif y < 2000:
                temel_egilim[i] = 0.15 + 0.012 * (y - 1970)
            else:
                temel_egilim[i] = 0.52 + 0.025 * (y - 2000)

        np.random.seed(42)
        degiskenlik = np.random.normal(0, 0.08, len(yillar))

        volkanik = np.zeros(len(yillar), dtype=float)
        volkanik_yillar = {
            1883: -0.2, 1902: -0.15, 1963: -0.1,
            1982: -0.15, 1991: -0.2
        }
        for vy, buyukluk in volkanik_yillar.items():
            idx = np.where(yillar == vy)[0]
            if len(idx) > 0:
                for j in range(3):
                    if idx[0] + j < len(yillar):
                        volkanik[idx[0] + j] = buyukluk * (1 - j / 3)

        anomali = temel_egilim + degiskenlik + volkanik
        return pd.DataFrame({
            'yil': yillar.astype(int),
            'sicaklik_anomalisi': np.round(anomali, 2).astype(float)
        })

    # ── CO₂ ──
    def co2_indir(self, durum):
        durum.update(label="NOAA CO₂ verileri indiriliyor…", state="running")
        try:
            yanit = requests.get(self.co2_url, timeout=15)
            yanit.raise_for_status()

            satirlar = yanit.text.split('\n')
            veri_satirlari = [
                s for s in satirlar if s.strip() and not s.startswith('#')
            ]
            veri_metni = '\n'.join(veri_satirlari)
            df = pd.read_csv(
                io.StringIO(veri_metni),
                names=['yil', 'co2', 'belirsizlik'],
                skipinitialspace=True
            )
            df = df[['yil', 'co2']].copy()
            df['yil'] = df['yil'].apply(self._guvenli_float)
            df['co2'] = df['co2'].apply(self._guvenli_float)
            df = df.dropna().reset_index(drop=True)
            df['yil'] = df['yil'].astype(int)
            df['co2'] = df['co2'].astype(float)

            durum.update(
                label="CO₂: {} yıl ({}-{})".format(
                    len(df), int(df['yil'].min()), int(df['yil'].max())
                ),
                state="complete"
            )
            return df

        except Exception:
            durum.update(
                label="CO₂ indirilemedi — yedek veri", state="complete"
            )
            return self._co2_yedek()

    def _co2_yedek(self):
        yillar = np.arange(1958, 2025)
        co2 = 315.0 + 1.3 * (yillar - 1958) + 0.013 * (yillar - 1958) ** 2
        return pd.DataFrame({
            'yil': yillar.astype(int),
            'co2': np.round(co2, 1).astype(float)
        })

    # ── KLOROFİL-a ──
    def klorofil_verisi_al(self, durum):
        durum.update(
            label="MODIS-Aqua Klorofil-a hazırlanıyor…", state="running"
        )
        yillar = np.arange(2002, 2025)
        klorofil_temel = np.array([
            0.283, 0.278, 0.275, 0.271, 0.268, 0.272, 0.265, 0.261,
            0.258, 0.262, 0.255, 0.252, 0.248, 0.251, 0.245, 0.242,
            0.238, 0.241, 0.235, 0.232, 0.229, 0.226, 0.223
        ])
        np.random.seed(123)
        degiskenlik = np.random.normal(0, 0.005, len(yillar))
        klorofil_a = klorofil_temel + degiskenlik

        df = pd.DataFrame({
            'yil': yillar.astype(int),
            'klorofil_a_ortalama': np.round(klorofil_a, 4).astype(float)
        })
        durum.update(
            label="Klorofil-a: {} yıl".format(len(df)), state="complete"
        )
        return df


# ══════════════════════════════════════════════════════════════════════
#  BÖLÜM 2 — FİTOPLANKTON MODELİ
# ══════════════════════════════════════════════════════════════════════

class FitoplanktonModeli:
    """
    Eppley büyüme eğrisi + Monod besin kısıtlaması
    + termal stratifikasyon etkisi.
    """

    def __init__(self, T_opt=20.0, strat_hass=0.15, K_N=0.5):
        self.mu_ref = 0.59            # referans büyüme hızı (gün⁻¹)
        self.T_opt = float(T_opt)     # optimal sıcaklık (°C)
        self.T_maks = 35.0            # üst sınır (°C)
        self.T_min = -2.0             # alt sınır (°C)
        self.K_N = float(K_N)         # yarı doygunluk sabiti
        self.N0 = 5.0                 # temel besin konsantrasyonu
        self.strat_hassasiyet = float(strat_hass)

    def eppley_buyume_hizi(self, T):
        """Eppley (1972) sıcaklık-büyüme ilişkisi."""
        return self.mu_ref * np.exp(0.0633 * float(T))

    def termal_pencere(self, T):
        """Sıcaklığın optimal aralığa göre büyüme kısıtlaması (0-1)."""
        T = float(T)
        if T <= self.T_min or T >= self.T_maks:
            return 0.0
        if T <= self.T_opt:
            t_norm = (T - self.T_min) / (self.T_opt - self.T_min)
        else:
            t_norm = (self.T_maks - T) / (self.T_maks - self.T_opt)
        t_norm = max(0.0, min(1.0, t_norm))
        return t_norm ** 1.5

    def stratifikasyon_etkisi(self, delta_T):
        """Sıcaklık artışının dikey karışımı zayıflatma etkisi."""
        return np.exp(-self.strat_hassasiyet * max(float(delta_T), 0.0))

    def besin_kisitlamasi(self, N):
        """Monod kinetik besin kısıtlaması (0-1)."""
        return float(N) / (self.K_N + float(N))

    def populasyon_hesapla(self, T_serisi, delta_T_serisi, yillar):
        """Yıllık fitoplankton popülasyon indeksi hesaplar."""
        net_buyumeler = []
        buyume_hizlari = []
        besin_faktorleri = []

        for i in range(len(yillar)):
            T = float(T_serisi[i])
            dT = float(delta_T_serisi[i])

            mu_maks = self.eppley_buyume_hizi(T)
            tp = self.termal_pencere(T)
            strat = self.stratifikasyon_etkisi(dT)
            N_mevcut = self.N0 * strat
            besin_lim = self.besin_kisitlamasi(N_mevcut)

            mu_etkin = mu_maks * tp * besin_lim
            kayip_hizi = 0.1 + 0.005 * max(dT, 0.0)
            net = mu_etkin - kayip_hizi

            buyume_hizlari.append(float(mu_etkin))
            besin_faktorleri.append(float(besin_lim))
            net_buyumeler.append(float(net))

        net_buyumeler = np.array(net_buyumeler, dtype=float)
        buyume_hizlari = np.array(buyume_hizlari, dtype=float)
        besin_faktorleri = np.array(besin_faktorleri, dtype=float)

        # Kümülatif indeks oluştur
        erken_ort = float(np.mean(net_buyumeler[:min(30, len(net_buyumeler))]))
        kumulatif = np.cumsum(net_buyumeler - erken_ort)
        pop_relatif = 1.0 + 0.01 * kumulatif

        # 1950 yılını referans al (= 1.0)
        yillar_arr = np.array(yillar, dtype=float)
        idx_1950 = int(np.argmin(np.abs(yillar_arr - 1950.0)))
        if abs(pop_relatif[idx_1950]) > 1e-10:
            pop_relatif = pop_relatif / pop_relatif[idx_1950]

        return pop_relatif, buyume_hizlari, besin_faktorleri


# ══════════════════════════════════════════════════════════════════════
#  BÖLÜM 3 — DENİZ BESİN ZİNCİRİ  (Lotka-Volterra, 5 trofik seviye)
# ══════════════════════════════════════════════════════════════════════

class DenizBesinZinciri:
    """
    Beş trofik seviyeli Lotka-Volterra ODE sistemi.
    Sıra: Fitoplankton → Zooplankton → Küçük Balıklar
          → Büyük Balıklar → Üst Yırtıcılar
    """

    def __init__(self):
        self.baslangic_pop = np.array(
            [1.0, 0.5, 0.3, 0.15, 0.05], dtype=float
        )
        self.olum_hizlari = np.array(
            [0.05, 0.15, 0.2, 0.25, 0.1], dtype=float
        )
        self.avlanma_ver = np.array(
            [0.0, 0.4, 0.3, 0.2, 0.15], dtype=float
        )
        self.donusum_ver = np.array(
            [1.0, 0.15, 0.12, 0.10, 0.08], dtype=float
        )
        self.sicaklik_hass = np.array(
            [0.8, 0.5, 0.4, 0.3, 0.2], dtype=float
        )

    def besin_zinciri_ode(self, y, t, fito_zorlama, sicaklik_stresi):
        """5-bileşenli ODE sistemi."""
        P = max(float(y[0]), 0.001)   # Fitoplankton
        Z = max(float(y[1]), 0.001)   # Zooplankton
        K = max(float(y[2]), 0.001)   # Küçük balıklar
        B = max(float(y[3]), 0.001)   # Büyük balıklar
        U = max(float(y[4]), 0.001)   # Üst yırtıcılar

        fz = float(fito_zorlama)
        ss = float(sicaklik_stresi)

        dP = (fz * P * (1.0 - P / 1.5)
              - self.avlanma_ver[1] * P * Z)

        dZ = (self.donusum_ver[1] * self.avlanma_ver[1] * P * Z
              - self.olum_hizlari[1] * Z
              - self.avlanma_ver[2] * Z * K
              - ss * self.sicaklik_hass[1] * Z)

        dK = (self.donusum_ver[2] * self.avlanma_ver[2] * Z * K
              - self.olum_hizlari[2] * K
              - self.avlanma_ver[3] * K * B
              - ss * self.sicaklik_hass[2] * K)

        dB = (self.donusum_ver[3] * self.avlanma_ver[3] * K * B
              - self.olum_hizlari[3] * B
              - self.avlanma_ver[4] * B * U
              - ss * self.sicaklik_hass[3] * B)

        dU = (self.donusum_ver[4] * self.avlanma_ver[4] * B * U
              - self.olum_hizlari[4] * U
              - ss * self.sicaklik_hass[4] * U)

        return [dP, dZ, dK, dB, dU]

    def kaskad_simulasyonu(self, fito_populasyonu, sicaklik_anomalileri,
                           yillar):
        """Besin zinciri boyunca kaskad etkisini simüle eder."""
        n = len(yillar)
        populasyonlar = np.zeros((n, 5), dtype=float)
        populasyonlar[0] = self.baslangic_pop.copy()

        for i in range(1, n):
            fito_degisim = (
                float(fito_populasyonu[i])
                - float(fito_populasyonu[i - 1])
            )
            fito_zorlama = 0.5 * (1.0 + fito_degisim)
            sicaklik_stresi = max(
                0.0, float(sicaklik_anomalileri[i]) * 0.02
            )

            t_aralik = np.linspace(0.0, 1.0, 50)
            y0 = np.maximum(populasyonlar[i - 1].copy(), 0.001)

            try:
                cozum = odeint(
                    self.besin_zinciri_ode, y0, t_aralik,
                    args=(fito_zorlama, sicaklik_stresi),
                    mxstep=5000
                )
                populasyonlar[i] = np.maximum(cozum[-1], 0.001)
            except Exception:
                populasyonlar[i] = populasyonlar[i - 1].copy()

            # Fitoplankton katmanını doğrudan modelden zorla
            populasyonlar[i, 0] = max(float(fito_populasyonu[i]), 0.01)

        # Başlangıca göre normalize et
        for j in range(5):
            if populasyonlar[0, j] > 0:
                populasyonlar[:, j] /= populasyonlar[0, j]

        return populasyonlar


# ══════════════════════════════════════════════════════════════════════
#  BÖLÜM 4 — İKLİM SENARYOLARI  (IPCC AR6 — SSP)
# ══════════════════════════════════════════════════════════════════════

class IklimSenaryolari:
    """IPCC AR6 SSP senaryolarına dayalı projeksiyon motoru."""

    def __init__(self):
        self.senaryolar = {
            'SSP1-2.6': {
                'renk': '#2ecc71',
                'etiket': 'SSP1-2.6 (Düşük Emisyon)',
                'aciklama': 'Paris hedefleri karşılanır',
                'sicaklik_2100': 1.8,
                'zirve_yili': 2040,
            },
            'SSP2-4.5': {
                'renk': '#f39c12',
                'etiket': 'SSP2-4.5 (Orta Senaryo)',
                'aciklama': 'Mevcut politikalar devam eder',
                'sicaklik_2100': 2.7,
                'zirve_yili': 2080,
            },
            'SSP5-8.5': {
                'renk': '#e74c3c',
                'etiket': 'SSP5-8.5 (Yüksek Emisyon)',
                'aciklama': 'Fosil yakıt bağımlılığı sürer',
                'sicaklik_2100': 4.4,
                'zirve_yili': None,
            }
        }

    def sicaklik_projeksiyonu(self, mevcut_yil, mevcut_sicaklik,
                              hedef_yil=2100):
        """Her SSP senaryosu için sıcaklık yolu üretir."""
        gelecek_yillar = np.arange(mevcut_yil, hedef_yil + 1)
        projeksiyonlar = {}

        for isim, prm in self.senaryolar.items():
            n = len(gelecek_yillar)
            artis = prm['sicaklik_2100'] - float(mevcut_sicaklik)

            if prm['zirve_yili'] and prm['zirve_yili'] < hedef_yil:
                sicakliklar = np.zeros(n, dtype=float)
                for i, y in enumerate(gelecek_yillar):
                    if y <= prm['zirve_yili']:
                        ilerleme = (
                            (y - mevcut_yil)
                            / (prm['zirve_yili'] - mevcut_yil)
                        )
                        sicakliklar[i] = (
                            float(mevcut_sicaklik) + artis * ilerleme
                        )
                    else:
                        dusus = (
                            0.3
                            * (y - prm['zirve_yili'])
                            / (hedef_yil - prm['zirve_yili'])
                        )
                        sicakliklar[i] = (
                            float(mevcut_sicaklik)
                            + artis
                            - artis * dusus * 0.2
                        )
            else:
                ilerleme = (
                    (gelecek_yillar - mevcut_yil).astype(float)
                    / float(hedef_yil - mevcut_yil)
                )
                sicakliklar = (
                    float(mevcut_sicaklik) + artis * ilerleme ** 1.2
                )

            np.random.seed(abs(hash(isim)) % (2 ** 31))
            gurultu = np.random.normal(0, 0.05, n)
            sicakliklar = sicakliklar + gurultu

            projeksiyonlar[isim] = {
                'yillar': gelecek_yillar,
                'sicakliklar': sicakliklar.astype(float),
                'parametreler': prm
            }

        return projeksiyonlar

    def fitoplankton_projeksiyonu(self, projeksiyonlar, fito_modeli):
        """Her senaryo için fitoplankton popülasyonu hesaplar."""
        fito_projeksiyonlar = {}
        temel_dyo = 17.0  # temel deniz yüzey sıcaklığı

        for isim, proj in projeksiyonlar.items():
            sicakliklar = proj['sicakliklar']
            dyo = temel_dyo + sicakliklar
            delta_T = sicakliklar
            pop, _, _ = fito_modeli.populasyon_hesapla(
                dyo, delta_T, proj['yillar']
            )

            fito_projeksiyonlar[isim] = {
                'yillar': proj['yillar'],
                'populasyon': pop.astype(float),
                'parametreler': proj['parametreler']
            }

        return fito_projeksiyonlar


# ══════════════════════════════════════════════════════════════════════
#  BÖLÜM 5 — GRAFİK FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════

def grafik_sicaklik(yillar, anomali):
    """Bar + polinom trend grafiği — sıcaklık anomalisi."""
    fig, ax = plt.subplots(figsize=(14, 5))
    renkler = ['#3498db' if t < 0 else '#e74c3c' for t in anomali]
    ax.bar(yillar, anomali, color=renkler, alpha=0.7, width=1.0)

    z = np.polyfit(yillar, anomali, 3)
    p = np.poly1d(z)
    ax.plot(yillar, p(yillar), 'k-', linewidth=2.5,
            label='Polinom Eğilim')

    ax.set_title('Küresel Sıcaklık Anomalisi (NASA GISTEMP v4)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Yıl')
    ax.set_ylabel('Sıcaklık Anomalisi (°C)')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    son_sicaklik = float(anomali[-1])
    son_yil = float(yillar[-1])
    ax.annotate(
        '{:.2f} °C'.format(son_sicaklik),
        xy=(son_yil, son_sicaklik),
        xytext=(-60, 20), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=12, fontweight='bold', color='red'
    )
    plt.tight_layout()
    return fig


def grafik_co2(co2_df):
    """Alan + çizgi grafiği — atmosferik CO₂."""
    fig, ax = plt.subplots(figsize=(14, 5))
    yillar = np.array(co2_df['yil'], dtype=float)
    degerler = np.array(co2_df['co2'], dtype=float)

    ax.fill_between(yillar, 280, degerler, alpha=0.3, color='#e74c3c')
    ax.plot(yillar, degerler, color='#c0392b', linewidth=2)
    ax.axhline(y=280, color='green', linewidth=1, linestyle='--',
               label='Sanayi öncesi (280 ppm)')
    ax.axhline(y=350, color='orange', linewidth=1, linestyle='--',
               label='Güvenli sınır (350 ppm)')

    ax.set_title('Atmosferik CO₂ Konsantrasyonu (Mauna Loa)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Yıl')
    ax.set_ylabel('CO₂ (ppm)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    son_co2 = float(degerler[-1])
    ax.annotate(
        '{:.1f} ppm'.format(son_co2),
        xy=(float(yillar[-1]), son_co2),
        xytext=(-80, -30), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=12, fontweight='bold', color='#c0392b'
    )
    plt.tight_layout()
    return fig


def grafik_fitoplankton(yillar, fito):
    """Alan grafiği — fitoplankton popülasyon indeksi."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(yillar, 1.0, fito, where=(fito >= 1.0),
                    alpha=0.3, color='green', label='Artış')
    ax.fill_between(yillar, 1.0, fito, where=(fito < 1.0),
                    alpha=0.3, color='red', label='Azalış')
    ax.plot(yillar, fito, color='#27ae60', linewidth=2)
    ax.axhline(y=1.0, color='black', linewidth=1, linestyle='--')

    ax.set_title('Fitoplankton Popülasyon İndeksi',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Yıl')
    ax.set_ylabel('Göreli Popülasyon (1950 = 1,0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    yuzde = (float(fito[-1]) / float(fito[0]) - 1.0) * 100.0
    ax.text(
        0.02, 0.05,
        'Toplam değişim: {:+.1f} %'.format(yuzde),
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        color='red' if yuzde < 0 else 'green',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    plt.tight_layout()
    return fig


def grafik_stratifikasyon(yillar, besin, buyume):
    """İki eksenli grafik — besin erişilebilirliği & büyüme hızı."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax2 = ax.twinx()

    c1, = ax.plot(yillar, besin, color='#3498db', linewidth=2,
                  label='Besin Erişilebilirliği')
    c2, = ax2.plot(yillar, buyume, color='#e67e22', linewidth=2,
                   label='Büyüme Hızı')

    ax.set_title('Stratifikasyon Etkisi: Besin ↔ Büyüme',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Yıl')
    ax.set_ylabel('Besin Erişilebilirliği (0–1)', color='#3498db')
    ax2.set_ylabel('Büyüme Hızı (gün⁻¹)', color='#e67e22')
    ax.legend([c1, c2], [c1.get_label(), c2.get_label()],
              loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def grafik_besin_zinciri(yillar, zincir_pop):
    """Beş trofik seviyenin zaman serisi."""
    fig, ax = plt.subplots(figsize=(14, 6))
    zincir_pop = np.array(zincir_pop, dtype=float)
    yillar = np.array(yillar, dtype=float)

    cizgi_stilleri = ['-', '-', '--', '--', ':']

    for j in range(5):
        ax.plot(yillar, zincir_pop[:, j],
                color=TROFIK_RENKLER[j], linewidth=2,
                linestyle=cizgi_stilleri[j],
                label=TROFIK_ISIMLER[j])

    ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--',
               alpha=0.5)
    ax.fill_between(yillar, 0.95, 1.05, alpha=0.08, color='green',
                    label='Kararlı bölge')

    ax.set_title('Besin Zinciri Kaskad Etkisi — 5 Trofik Seviye',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Yıl', fontsize=12)
    ax.set_ylabel('Göreli Popülasyon (başlangıç = 1,0)', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    for j in range(5):
        son = float(zincir_pop[-1, j])
        degisim = (son - 1.0) * 100.0
        ax.annotate(
            '{:+.1f} %'.format(degisim),
            xy=(float(yillar[-1]), son),
            xytext=(8, 0), textcoords='offset points',
            fontsize=9, fontweight='bold',
            color='red' if degisim < 0 else 'green'
        )

    plt.tight_layout()
    return fig


def grafik_ekolojik_piramit(zincir_pop, yillar):
    """
    Sol panel : Ekolojik piramit (başlangıç ↔ güncel)
    Sağ panel : Kümülatif yüzde değişim zaman serisi
    """
    fig, eksenler = plt.subplots(1, 2, figsize=(18, 8))
    zincir_pop = np.array(zincir_pop, dtype=float)
    yillar = np.array(yillar, dtype=float)

    # ── SOL: PİRAMİT ──
    ax = eksenler[0]
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Ekolojik Piramit (Başlangıç ↔ Güncel)',
                 fontsize=13, fontweight='bold')

    # alttan üste: i=0 en geniş (fitoplankton), i=4 en dar (üst yırtıcı)
    genislikler = [2.5, 2.0, 1.5, 1.0, 0.5]

    baslangic = zincir_pop[0]
    guncel = zincir_pop[-1]

    for i in range(5):
        y = float(i)
        w = genislikler[i]

        # Başlangıç (sol, soluk)
        ax.add_patch(plt.Rectangle(
            (-w, y - 0.15), w, 0.3,
            color=TROFIK_RENKLER[i], alpha=0.4
        ))

        # Güncel (sağ, koyu)
        sp = float(baslangic[i])
        ep = float(guncel[i])
        oran = ep / sp if sp > 0.001 else 1.0
        yeni_w = max(min(w * oran, 2.8), 0.05)

        ax.add_patch(plt.Rectangle(
            (0, y - 0.15), yeni_w, 0.3,
            color=TROFIK_RENKLER[i], alpha=0.8
        ))

        degisim = (oran - 1.0) * 100.0

        ax.text(-w - 0.1, y, TROFIK_ISIMLER[i],
                ha='right', va='center', fontsize=9, fontweight='bold')

        metin_x = max(yeni_w, w) + 0.1
        ax.text(metin_x, y, '{:+.1f} %'.format(degisim),
                ha='left', va='center', fontsize=11, fontweight='bold',
                color='red' if degisim < 0 else 'green')

    ax.text(-1.25, 5.2, 'Başlangıç', ha='center', fontsize=10,
            color='gray')
    ax.text(1.25, 5.2, 'Güncel', ha='center', fontsize=10,
            fontweight='bold')

    # ── SAĞ: KÜMÜLATİF DEĞİŞİM ──
    ax2 = eksenler[1]
    for j in range(5):
        taban = float(zincir_pop[0, j])
        if taban > 0:
            yuzde = (zincir_pop[:, j] / taban - 1.0) * 100.0
        else:
            yuzde = np.zeros(len(yillar))
        ax2.plot(yillar, yuzde, color=TROFIK_RENKLER[j], linewidth=2,
                 label=TROFIK_ISIMLER[j])

    ax2.axhline(y=0, color='black', linewidth=1, linestyle='--')
    ax2.fill_between(yillar, -5, 5, alpha=0.1, color='green',
                     label='Kararlı bölge')
    ax2.set_title('Kümülatif Değişim (%)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Yıl')
    ax2.set_ylabel('Popülasyon Değişimi (%)')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def grafik_projeksiyonlar(yillar, fito, projeksiyonlar, mevcut_yil):
    """Tarihsel veri + üç SSP senaryosu."""
    fig, ax = plt.subplots(figsize=(16, 6))

    for isim, proj in projeksiyonlar.items():
        prm = proj['parametreler']
        py = np.array(proj['yillar'], dtype=float)
        pp = np.array(proj['populasyon'], dtype=float)
        ax.plot(py, pp, color=prm['renk'], linewidth=2.5,
                label=prm['etiket'])
        ax.fill_between(py, pp, alpha=0.1, color=prm['renk'])
        son_deger = float(pp[-1])
        ax.annotate(
            '{:.2f}'.format(son_deger),
            xy=(float(py[-1]), son_deger),
            xytext=(10, 0), textcoords='offset points',
            fontsize=10, fontweight='bold', color=prm['renk']
        )

    maske = yillar <= mevcut_yil
    ax.plot(yillar[maske], fito[maske], color='black', linewidth=2,
            label='Gözlenen (model)')
    ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--')
    ax.axvline(x=mevcut_yil, color='gray', linewidth=1,
               linestyle=':', alpha=0.7)
    ax.text(mevcut_yil + 1, ax.get_ylim()[1] * 0.95, 'Bugün',
            fontsize=10, color='gray')

    ax.set_title(
        'IPCC Senaryoları: Fitoplankton Projeksiyonları (2024–2100)',
        fontsize=14, fontweight='bold')
    ax.set_xlabel('Yıl', fontsize=12)
    ax.set_ylabel('Göreli Popülasyon', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1950, 2105)
    plt.tight_layout()
    return fig


def grafik_mekanizma():
    """Neden-sonuç akış diyagramı."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(
        'KÜRESEL ISINMA → FİTOPLANKTON → BESİN ZİNCİRİ\n'
        'Etki Mekanizması',
        fontsize=16, fontweight='bold', pad=20
    )

    kutular = [
        (1,  8, 3,   1.2, 'CO₂ Emisyonları\nArtışı',         '#e74c3c'),
        (6,  8, 3.5, 1.2, 'Küresel Sıcaklık\nArtışı (+1,2 °C)', '#e67e22'),
        (12, 8, 3,   1.2, 'Okyanus\nIsınması',                '#3498db'),
        (1,  5, 3.5, 1.5, 'Termal\nStratifikasyon\nGüçlenmesi', '#9b59b6'),
        (6,  5, 3.5, 1.5, 'Besin Tuzu\nTaşınmasında\nAzalma', '#c0392b'),
        (12, 5, 3,   1.5, 'Fitoplankton\nPopülasyonu\nAzalma', '#27ae60'),
        (1,  1.5, 3, 1.5, 'Zooplankton\nAzalma',              '#3498db'),
        (5.5,1.5, 3, 1.5, 'Balık Stokları\nAzalma',           '#9b59b6'),
        (10, 1.5, 3, 1.5, 'O₂ Üretimi\nAzalma',              '#e74c3c'),
        (14, 1.5,1.8,1.5, 'Ekosistem\nÇöküşü',               '#c0392b'),
    ]

    for (x, y, w, h, metin, renk) in kutular:
        ax.add_patch(plt.Rectangle(
            (x, y), w, h, linewidth=2,
            edgecolor=renk, facecolor=renk, alpha=0.15
        ))
        ax.add_patch(plt.Rectangle(
            (x, y), w, h, linewidth=2,
            edgecolor=renk, facecolor='none'
        ))
        ax.text(x + w / 2, y + h / 2, metin,
                ha='center', va='center', fontsize=9, fontweight='bold')

    # Ok yönleri: FROM (x1,y1) TO (x2,y2)
    oklar = [
        (4,    8.6,  6,    8.6),    # CO₂ → Sıcaklık
        (9.5,  8.6,  12,   8.6),    # Sıcaklık → Okyanus
        (13.5, 8.0,  13.5, 6.5),    # Okyanus ↓ Fitoplankton
        (12.5, 8.0,  2.75, 6.5),    # Okyanus → Stratifikasyon
        (4.5,  5.75, 6.0,  5.75),   # Stratifikasyon → Besin
        (9.5,  5.75, 12.0, 5.75),   # Besin → Fitoplankton
        (2.5,  5.0,  2.5,  3.0),    # Stratifikasyon ↓ Zooplankton
        (7.75, 5.0,  7.0,  3.0),    # Besin ↓ Balık Stokları
        (13.5, 5.0,  11.5, 3.0),    # Fitoplankton ↓ O₂
        (8.5,  2.25, 10.0, 2.25),   # Balık → O₂
        (13.0, 2.25, 14.0, 2.25),   # O₂ → Çöküş
    ]

    for (x1, y1, x2, y2) in oklar:
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->', color='gray', lw=2,
                connectionstyle='arc3,rad=0.1'
            )
        )

    plt.tight_layout()
    return fig


def grafik_senaryo_zincirleri(projeksiyonlar, besin_zinciri_modeli):
    """2×2 panel: 3 SSP senaryosu + karşılaştırma."""
    fig, eksenler = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        'IPCC Senaryoları: Besin Zinciri Projeksiyonları',
        fontsize=16, fontweight='bold'
    )

    senaryo_listesi = list(projeksiyonlar.items())

    for idx in range(min(3, len(senaryo_listesi))):
        isim, proj = senaryo_listesi[idx]
        ax = eksenler[idx // 2, idx % 2]

        py = np.array(proj['yillar'], dtype=float)
        fp = np.array(proj['populasyon'], dtype=float)
        sa = np.linspace(
            0, proj['parametreler']['sicaklik_2100'] - 1.2, len(py)
        )

        zp = besin_zinciri_modeli.kaskad_simulasyonu(fp, sa, py)

        for j in range(5):
            ax.plot(py, zp[:, j], color=TROFIK_RENKLER[j],
                    linewidth=2, label=TROFIK_ISIMLER[j])

        ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(
            '{}\n({})'.format(
                proj['parametreler']['etiket'],
                proj['parametreler']['aciklama']
            ),
            fontsize=11, fontweight='bold',
            color=proj['parametreler']['renk']
        )
        ax.set_xlabel('Yıl')
        ax.set_ylabel('Göreli Popülasyon')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 2.0)

    # sağ-alt panel: fitoplankton karşılaştırma
    ax4 = eksenler[1, 1]
    for isim, proj in projeksiyonlar.items():
        prm = proj['parametreler']
        py = np.array(proj['yillar'], dtype=float)
        pp = np.array(proj['populasyon'], dtype=float)
        ax4.plot(py, pp, color=prm['renk'], linewidth=2.5,
                 label=prm['etiket'])
        ax4.fill_between(py, pp, alpha=0.1, color=prm['renk'])

    ax4.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
    ax4.set_title('Fitoplankton Karşılaştırma',
                  fontsize=11, fontweight='bold')
    ax4.set_xlabel('Yıl')
    ax4.set_ylabel('Göreli Popülasyon')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
#  BÖLÜM 6 — STREAMLIT ANA UYGULAMA
# ══════════════════════════════════════════════════════════════════════

def ana_uygulama():

    # ── BAŞLIK ──
    st.markdown("""
    # 🌊 Küresel Isınmanın Fitoplanktonlara Etkisi
    ## ve Besin Zinciri Kaskad Simülasyonu

    **Veri Kaynakları:** NASA GISTEMP v4 · NASA MODIS-Aqua
    · NOAA Mauna Loa · IPCC AR6

    ---
    """)

    # ── YAN PANEL ──
    st.sidebar.header("⚙️ Simülasyon Parametreleri")

    st.sidebar.subheader("Fitoplankton Modeli")
    T_opt = st.sidebar.slider(
        "Optimal Sıcaklık (°C)", 15.0, 30.0, 20.0, 0.5,
        help="Fitoplanktonun en iyi büyüdüğü sıcaklık"
    )
    strat_hass = st.sidebar.slider(
        "Stratifikasyon Hassasiyeti", 0.05, 0.40, 0.15, 0.01,
        help="Sıcaklık artışının besin taşınmasına etkisi"
    )
    K_N = st.sidebar.slider(
        "Yarı Doygunluk Sabiti (Monod K_N)", 0.1, 2.0, 0.5, 0.1,
        help="Besin kısıtlaması parametresi"
    )

    st.sidebar.subheader("Analiz Dönemi")
    baslangic_yili = st.sidebar.slider(
        "Başlangıç Yılı", 1880, 1980, 1880, 10
    )
    bitis_yili = st.sidebar.slider("Bitiş Yılı", 2000, 2024, 2024, 1)

    st.sidebar.subheader("Projeksiyon")
    proj_hedef = st.sidebar.slider(
        "Projeksiyon Hedef Yılı", 2050, 2150, 2100, 10
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Python {} | Simülasyon v3.1".format(sys.version.split()[0])
    )

    btn_baslat = st.sidebar.button(
        "🚀 Simülasyonu Başlat", type="primary", use_container_width=True
    )

    # ── BAŞLANGIÇ EKRANI ──
    if not btn_baslat and 'sonuclar' not in st.session_state:
        st.info(
            "👈 Sol paneldeki parametreleri ayarlayın ve "
            "**Simülasyonu Başlat** düğmesine basın."
        )

        with st.expander("📖 Simülasyon Hakkında", expanded=True):
            st.markdown("""
            ### Kullanılan Modeller

            | Model | Formül | Açıklama |
            |-------|--------|----------|
            | **Eppley Eğrisi** | `μ = 0,59 · e^(0,0633·T)` | Sıcaklık–büyüme |
            | **Monod Kinetiği** | `f(N) = N / (K + N)` | Besin kısıtlaması |
            | **Stratifikasyon** | `S = e^(−0,15·ΔT)` | Isınma → besin azalması |
            | **Lotka-Volterra** | 5 ODE sistemi | Besin zinciri dinamiği |
            | **IPCC SSP** | 3 senaryo | Gelecek projeksiyonları |

            ### Besin Zinciri (5 Trofik Seviye)
            ```
            Fitoplankton → Zooplankton → Küçük Balıklar
            → Büyük Balıklar → Üst Yırtıcılar
            ```

            ### Kaynaklar
            - Eppley, R.W. (1972) *Fishery Bulletin*
            - Boyce vd. (2010) *Nature*, 466, 591-596
            - Behrenfeld vd. (2006) *Nature*, 444, 752-755
            - IPCC AR6 WG1 (2021)
            """)
        return

    # ── SİMÜLASYON ──
    if btn_baslat or 'sonuclar' in st.session_state:

        if btn_baslat:
            # ▸ ADIM 1: VERİ TOPLAMA
            st.header("📡 Adım 1 — Veri Toplama")
            toplayici = NASAVeriToplayici()

            with st.status(
                "NASA ve NOAA verileri indiriliyor…", expanded=True
            ) as durum:
                gistemp_df = toplayici.gistemp_indir(durum)

            with st.status("CO₂ verileri…", expanded=False) as durum:
                co2_df = toplayici.co2_indir(durum)

            with st.status(
                "Klorofil-a verileri…", expanded=False
            ) as durum:
                klorofil_df = toplayici.klorofil_verisi_al(durum)

            st.success("✅ Tüm veriler başarıyla toplandı!")

            # ▸ ADIM 2: VERİ HAZIRLAMA
            st.header("🔧 Adım 2 — Veri Hazırlama")
            maske = (
                (gistemp_df['yil'] >= baslangic_yili)
                & (gistemp_df['yil'] <= bitis_yili)
            )
            analiz_df = gistemp_df[maske].copy().reset_index(drop=True)

            yillar = np.array(analiz_df['yil'].values, dtype=float)
            sicaklik_anomalisi = np.array(
                analiz_df['sicaklik_anomalisi'].values, dtype=float
            )
            temel_dyo = 17.0
            dyo = temel_dyo + sicaklik_anomalisi * 0.7

            s1, s2, s3 = st.columns(3)
            s1.metric(
                "Dönem",
                "{:.0f} – {:.0f}".format(yillar[0], yillar[-1])
            )
            s2.metric("Veri Noktası", "{}".format(len(yillar)))
            s3.metric(
                "Sıcaklık Artışı",
                "{:+.2f} °C".format(
                    float(sicaklik_anomalisi[-1])
                    - float(sicaklik_anomalisi[0])
                )
            )

            # ▸ ADIM 3: FİTOPLANKTON MODELİ
            st.header("🦠 Adım 3 — Fitoplankton Modeli")
            fito_modeli = FitoplanktonModeli(
                T_opt=T_opt, strat_hass=strat_hass, K_N=K_N
            )

            with st.spinner(
                "Eppley + Stratifikasyon + Monod hesaplanıyor…"
            ):
                fito_pop, buyume_hizlari, besin_faktorleri = (
                    fito_modeli.populasyon_hesapla(
                        dyo, sicaklik_anomalisi, yillar
                    )
                )

            fito_degisim = (
                (float(fito_pop[-1]) / float(fito_pop[0]) - 1.0) * 100.0
            )
            s1, s2, s3 = st.columns(3)
            s1.metric(
                "Başlangıç İndeksi",
                "{:.3f}".format(float(fito_pop[0]))
            )
            s2.metric(
                "Son İndeks",
                "{:.3f}".format(float(fito_pop[-1]))
            )
            s3.metric(
                "Toplam Değişim",
                "{:+.1f} %".format(fito_degisim),
                delta="{:+.1f} %".format(fito_degisim)
            )

            # ▸ ADIM 4: BESİN ZİNCİRİ
            st.header("🔗 Adım 4 — Besin Zinciri Simülasyonu")
            besin_zinciri = DenizBesinZinciri()

            with st.spinner(
                "Lotka-Volterra ODE çözülüyor (5 trofik seviye)…"
            ):
                zincir_populasyonlari = besin_zinciri.kaskad_simulasyonu(
                    fito_pop, sicaklik_anomalisi, yillar
                )

            zincir_sutunlar = st.columns(5)
            for j, (sut, isim, ikon) in enumerate(
                zip(zincir_sutunlar, TROFIK_ISIMLER, TROFIK_IKONLAR)
            ):
                b = float(zincir_populasyonlari[0, j])
                s = float(zincir_populasyonlari[-1, j])
                dgs = (s / b - 1.0) * 100.0 if b > 0 else 0.0
                sut.metric(
                    "{} {}".format(ikon, isim),
                    "{:.3f}".format(s),
                    "{:+.1f} %".format(dgs)
                )

            # ▸ ADIM 5: PROJEKSİYONLAR
            st.header("🔮 Adım 5 — IPCC Projeksiyonları")
            senaryolar = IklimSenaryolari()

            with st.spinner("SSP senaryoları hesaplanıyor…"):
                sicaklik_proj = senaryolar.sicaklik_projeksiyonu(
                    2024, float(sicaklik_anomalisi[-1]),
                    hedef_yil=proj_hedef
                )
                fito_proj = senaryolar.fitoplankton_projeksiyonu(
                    sicaklik_proj, fito_modeli
                )

            proj_sutunlar = st.columns(3)
            for idx, (isim, proj) in enumerate(fito_proj.items()):
                prm = proj['parametreler']
                son = float(proj['populasyon'][-1])
                dgs = (son - 1.0) * 100.0
                proj_sutunlar[idx].metric(
                    prm['etiket'],
                    "{:.3f}".format(son),
                    "{:+.1f} %".format(dgs)
                )

            # ▸ SONUÇLARI KAYDET
            st.session_state['sonuclar'] = {
                'yillar': yillar,
                'sicaklik_anomalisi': sicaklik_anomalisi,
                'fito_pop': fito_pop,
                'buyume_hizlari': buyume_hizlari,
                'besin_faktorleri': besin_faktorleri,
                'zincir_populasyonlari': zincir_populasyonlari,
                'fito_proj': fito_proj,
                'co2_df': co2_df,
                'klorofil_df': klorofil_df,
                'besin_zinciri': besin_zinciri,
            }

        # ══════════════════════════════════════════════════════════
        #  GRAFİKLER
        # ══════════════════════════════════════════════════════════
        sn = st.session_state['sonuclar']

        st.markdown("---")
        st.header("📊 Sonuçlar ve Grafikler")

        t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
            "🌡️ Sıcaklık",
            "🏭 CO₂",
            "🦠 Fitoplankton",
            "🧪 Stratifikasyon",
            "🔗 Besin Zinciri",
            "🔮 Projeksiyonlar",
            "📐 Ekolojik Piramit",
            "🔄 Mekanizma"
        ])

        # ── Sekme 1: Sıcaklık ──
        with t1:
            st.subheader("Küresel Sıcaklık Anomalisi (NASA GISTEMP v4)")
            fig = grafik_sicaklik(sn['yillar'], sn['sicaklik_anomalisi'])
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 Veriyi Görüntüle"):
                gosterim_df = pd.DataFrame({
                    'Yıl': sn['yillar'].astype(int),
                    'Anomali (°C)': np.round(sn['sicaklik_anomalisi'], 3)
                })
                st.dataframe(
                    gosterim_df, use_container_width=True, height=300
                )

        # ── Sekme 2: CO₂ ──
        with t2:
            st.subheader("Atmosferik CO₂ Konsantrasyonu")
            fig = grafik_co2(sn['co2_df'])
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 Veriyi Görüntüle"):
                co2_gosterim = sn['co2_df'].copy()
                co2_gosterim.columns = ['Yıl', 'CO₂ (ppm)']
                st.dataframe(
                    co2_gosterim, use_container_width=True, height=300
                )

        # ── Sekme 3: Fitoplankton ──
        with t3:
            st.subheader("Fitoplankton Popülasyon İndeksi")
            fig = grafik_fitoplankton(sn['yillar'], sn['fito_pop'])
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("""
            > **Yorum:** Sıcaklık artışı tek başına fitoplankton
            > büyümesini hızlandırabilir (Eppley eğrisi), ancak
            > stratifikasyon nedeniyle besin taşınmasının azalması
            > **net etkiyi olumsuz** kılmaktadır.
            """)

        # ── Sekme 4: Stratifikasyon ──
        with t4:
            st.subheader("Stratifikasyon Etkisi: Besin ↔ Büyüme")
            fig = grafik_stratifikasyon(
                sn['yillar'], sn['besin_faktorleri'],
                sn['buyume_hizlari']
            )
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("""
            > **Mavi:** Besin erişilebilirliği
            > (stratifikasyon artınca düşer)
            >
            > **Turuncu:** Büyüme hızı
            > (sıcaklıkla artar ama besin eksikliği sınırlar)
            """)

        # ── Sekme 5: Besin Zinciri ──
        with t5:
            st.subheader(
                "Besin Zinciri Kaskad Etkisi — 5 Trofik Seviye"
            )
            fig = grafik_besin_zinciri(
                sn['yillar'], sn['zincir_populasyonlari']
            )
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("""
            > **Kaskad Etkisi:** Fitoplankton popülasyonundaki
            > değişim, besin zincirinin her kademesinde gecikmeyle
            > ve büyüyerek hissedilir. Üst yırtıcılar en geç ama
            > en sert etkilenir.
            """)

            with st.expander("📋 Trofik Seviye Detayları"):
                zp = sn['zincir_populasyonlari']
                satirlar = []
                for j in range(5):
                    b = float(zp[0, j])
                    s = float(zp[-1, j])
                    d = (s / b - 1.0) * 100.0 if b > 0 else 0.0
                    satirlar.append({
                        'Trofik Seviye': TROFIK_ISIMLER[j],
                        'Başlangıç': round(b, 4),
                        'Son': round(s, 4),
                        'Değişim (%)': round(d, 1)
                    })
                st.dataframe(
                    pd.DataFrame(satirlar), use_container_width=True
                )

        # ── Sekme 6: Projeksiyonlar ──
        with t6:
            st.subheader(
                "IPCC Senaryolarına Göre Fitoplankton Projeksiyonları"
            )
            fig = grafik_projeksiyonlar(
                sn['yillar'], sn['fito_pop'],
                sn['fito_proj'], 2024
            )
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Senaryo Bazlı Besin Zinciri Projeksiyonları")
            fig2 = grafik_senaryo_zincirleri(
                sn['fito_proj'], sn['besin_zinciri']
            )
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown("""
            | Senaryo | 2100 Sıcaklık | Açıklama |
            |---------|---------------|----------|
            | **SSP1-2.6** | +1,8 °C | Paris hedefleri karşılanır |
            | **SSP2-4.5** | +2,7 °C | Mevcut politikalar devam eder |
            | **SSP5-8.5** | +4,4 °C | Fosil yakıta bağımlılık sürer |
            """)

        # ── Sekme 7: Ekolojik Piramit ──
        with t7:
            st.subheader("Ekolojik Piramit ve Kümülatif Değişim")
            fig = grafik_ekolojik_piramit(
                sn['zincir_populasyonlari'], sn['yillar']
            )
            st.pyplot(fig)
            plt.close(fig)

        # ── Sekme 8: Mekanizma ──
        with t8:
            st.subheader("Etki Mekanizması Diyagramı")
            fig = grafik_mekanizma()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("""
            ```
            CO₂ Artışı
                │
                ▼
            Küresel Sıcaklık Artışı (+1,2 °C)
                │
                ▼
            Okyanus Isınması
                │
                ├──→ Termal Stratifikasyon Güçlenmesi
                │         │
                │         ▼
                │     Besin Tuzu Taşınması Azalır
                │         │
                │         ▼
                └──→ FİTOPLANKTON POPÜLASYONU AZALIR
                          │
                ┌─────────┼─────────┐
                │         │         │
                ▼         ▼         ▼
            Zooplankton  Balık    O₂ Üretimi
            Azalır     Stokları    Azalır
                       Azalır
            ```
            """)

        # ══════════════════════════════════════════════════════════
        #  ÖZET RAPOR
        # ══════════════════════════════════════════════════════════
        st.markdown("---")
        st.header("📋 Özet Rapor")

        sicaklik = sn['sicaklik_anomalisi']
        fito = sn['fito_pop']
        zincir = sn['zincir_populasyonlari']
        yillar = sn['yillar']

        rapor_sut = st.columns(2)

        with rapor_sut[0]:
            st.subheader("Gözlenen Değişimler")
            st.markdown("""
            - **Dönem:** {:.0f} – {:.0f}
            - **Sıcaklık:** {:.2f} °C → {:.2f} °C
              (Δ = {:+.2f} °C)
            - **Fitoplankton:** {:+.1f} % değişim
            """.format(
                yillar[0], yillar[-1],
                float(sicaklik[0]), float(sicaklik[-1]),
                float(sicaklik[-1]) - float(sicaklik[0]),
                (float(fito[-1]) / float(fito[0]) - 1) * 100
            ))

        with rapor_sut[1]:
            st.subheader("Besin Zinciri Etkileri")
            for j in range(5):
                b = float(zincir[0, j])
                s = float(zincir[-1, j])
                d = (s / b - 1.0) * 100.0 if b > 0 else 0.0
                ikon = '🔴' if d < 0 else '🟢'
                st.markdown(
                    "{} **{}**: {:+.1f} %".format(
                        ikon, TROFIK_ISIMLER[j], d
                    )
                )

        st.markdown("---")

        with st.expander("📚 Bilimsel Referanslar"):
            st.markdown("""
            1. **Eppley, R.W.** (1972) Temperature and phytoplankton
               growth. *Fishery Bulletin*, 70(4), 1063-1085
            2. **Boyce, D.G. vd.** (2010) Global phytoplankton decline.
               *Nature*, 466, 591-596
            3. **Behrenfeld, M.J. vd.** (2006) Climate-driven trends
               in ocean productivity. *Nature*, 444, 752-755
            4. **Henson, S.A. vd.** (2010) Detection of anthropogenic
               climate change. *Biogeosciences*, 7, 621-640
            5. **IPCC AR6 WG1** (2021) Climate Change:
               The Physical Science Basis
            6. **NASA GISTEMP v4:**
               https://data.giss.nasa.gov/gistemp
            7. **NASA Ocean Color:**
               https://oceancolor.gsfc.nasa.gov
            """)

        # ══════════════════════════════════════════════════════════
        #  CSV İNDİR
        # ══════════════════════════════════════════════════════════
        st.markdown("---")
        st.header("⬇️ Veri İndir")

        ind_sut = st.columns(3)

        with ind_sut[0]:
            sicaklik_csv = pd.DataFrame({
                'Yıl': yillar.astype(int),
                'Sıcaklık_Anomalisi_C': np.round(sicaklik, 3),
                'Fitoplankton_İndeksi': np.round(fito, 4)
            })
            st.download_button(
                "📥 Sıcaklık + Fitoplankton CSV",
                sicaklik_csv.to_csv(index=False).encode('utf-8'),
                "sicaklik_fitoplankton.csv",
                "text/csv"
            )

        with ind_sut[1]:
            zincir_csv_veri = {'Yıl': yillar.astype(int)}
            for j in range(5):
                zincir_csv_veri[TROFIK_ISIMLER[j]] = np.round(
                    zincir[:, j], 4
                )
            zincir_csv = pd.DataFrame(zincir_csv_veri)
            st.download_button(
                "📥 Besin Zinciri CSV",
                zincir_csv.to_csv(index=False).encode('utf-8'),
                "besin_zinciri.csv",
                "text/csv"
            )

        with ind_sut[2]:
            proj_satirlar = []
            for isim, proj in sn['fito_proj'].items():
                for i, y in enumerate(proj['yillar']):
                    proj_satirlar.append({
                        'Senaryo': isim,
                        'Yıl': int(y),
                        'Fitoplankton_İndeksi': round(
                            float(proj['populasyon'][i]), 4
                        )
                    })
            proj_csv = pd.DataFrame(proj_satirlar)
            st.download_button(
                "📥 Projeksiyon CSV",
                proj_csv.to_csv(index=False).encode('utf-8'),
                "projeksiyon.csv",
                "text/csv"
            )


# ══════════════════════════════════════════════════════════════════════
#  ÇALIŞTIR
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ana_uygulama()