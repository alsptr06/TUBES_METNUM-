import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ModelKualitasUdara:
    def __init__(self, df_pm25, df_co, df_o3):
        """
        Tahap 1: Membaca dan Menggabungkan 3 Dataset India (PM2.5, CO, O3)
        """
        print("Sedang memproses dan menggabungkan dataset...")
        
        # --- PERBAIKAN BULLETPROOF ---
        # Memotong DataFrame murni berdasarkan indeks kolom (Kolom ke-6 untuk Waktu, ke-3 untuk Nilai)
        # Ini mencegah KeyError jika Excel membaca header kolom secara aneh
        df_pm25 = df_pm25.iloc[:, [6, 3]].copy()
        df_pm25.columns = ['Waktu', 'PM25']
        
        df_co = df_co.iloc[:, [6, 3]].copy()
        df_co.columns = ['Waktu', 'CO']
        
        df_o3 = df_o3.iloc[:, [6, 3]].copy()  
        df_o3.columns = ['Waktu', 'O3']
        # -----------------------------
        
        # Proses penggabungan (Merge) ketiga data berdasarkan titik Waktu yang sama
        df_gabungan = pd.merge(df_pm25, df_co, on='Waktu', how='outer')
        df_gabungan = pd.merge(df_gabungan, df_o3, on='Waktu', how='outer')
        
        # Konversi ke format Datetime dan urutkan dari waktu paling awal
        df_gabungan['Waktu'] = pd.to_datetime(df_gabungan['Waktu'], utc=True)
        df_gabungan = df_gabungan.sort_values('Waktu').reset_index(drop=True)
        
        # Interpolasi untuk menambal data jika ada jam yang sensornya mati
        df_gabungan['PM25'] = pd.to_numeric(df_gabungan['PM25'], errors='coerce').interpolate(method='linear')
        df_gabungan['CO'] = pd.to_numeric(df_gabungan['CO'], errors='coerce').interpolate(method='linear')
        df_gabungan['O3'] = pd.to_numeric(df_gabungan['O3'], errors='coerce').interpolate(method='linear')
        
        # Menyimpan data ke dalam properti objek
        self.df = df_gabungan
        self.waktu = self.df['Waktu']
        self.P_aktual = self.df['PM25'].values  # Target perhitungan utama kita
        self.n_data = len(self.P_aktual)
        
        # PENENTUAN DELTA T (Interval Waktu Data India)
        # Data India direkam per 15 menit. Dalam satuan jam: 15/60 = 0.25 jam
        self.h = 0.25 

    def persamaan_diferensial(self, P, E, k):
        """
        PENYESUAIAN RUMUS
        Bagian ini adalah representasi dP/dt. 
        """
        return E - (k * P)

    def hitung_runge_kutta(self, E, k):
        """
        Tahap 2: Simulasi Peluruhan PM2.5 menggunakan metode numerik RK4
        """
        P_simulasi = np.zeros(self.n_data)
        P_simulasi[0] = self.P_aktual[0] # Titik awal diambil dari observasi pertama
        
        for i in range(self.n_data - 1):
            P_i = P_simulasi[i]
            
            # 4 Evaluasi K sesuai algoritma standar Runge-Kutta Orde 4
            K1 = self.persamaan_diferensial(P_i, E, k)
            K2 = self.persamaan_diferensial(P_i + (self.h / 2) * K1, E, k)
            K3 = self.persamaan_diferensial(P_i + (self.h / 2) * K2, E, k)
            K4 = self.persamaan_diferensial(P_i + self.h * K3, E, k)
            
            # Menghitung prediksi konsentrasi 1 langkah waktu (15 menit) ke depan
            P_simulasi[i+1] = P_i + (self.h / 6) * (K1 + 2*K2 + 2*K3 + K4)
            
        return P_simulasi

    def hitung_beda_pusat(self):
        """
        Menghitung akselerasi laju perubahan aktual per 15 menit menggunakan array data historis
        """
        laju_aktual = np.zeros(self.n_data)
        
        # Proses iterasi Turunan Numerik Beda Pusat
        for i in range(1, self.n_data - 1):
            laju_aktual[i] = (self.P_aktual[i+1] - self.P_aktual[i-1]) / (2 * self.h)
            
        # Penanganan ujung array awal dan akhir
        laju_aktual[0] = (self.P_aktual[1] - self.P_aktual[0]) / self.h
        laju_aktual[-1] = (self.P_aktual[-1] - self.P_aktual[-2]) / self.h
        
        return laju_aktual

    def deteksi_anomali(self, ambang_batas):
        laju_aktual = self.hitung_beda_pusat()
        
        # Identifikasi indeks mana saja yang lajunya melampaui batas kewajaran
        indeks_anomali = np.where(laju_aktual > ambang_batas)[0]
        
        print(f"\n--- DETEKSI KRISIS LOKAL (Lonjakan > {ambang_batas} µg/m³/jam) ---")
        if len(indeks_anomali) == 0:
            print("Kondisi stabil. Tidak ditemukan lonjakan ekstrem pada array.")
        else:
            print(f"Ditemukan {len(indeks_anomali)} titik anomali polusi:")
            # Hanya menampilkan 5 sampel pertama agar console tidak penuh
            for idx in indeks_anomali[:5]: 
                waktu_kejadian = self.waktu.iloc[idx].strftime('%Y-%m-%d %H:%M')
                print(f"-> {waktu_kejadian} | Akselerasi: +{laju_aktual[idx]:.2f} | PM2.5 Aktual: {self.P_aktual[idx]:.1f}")
                
        return indeks_anomali, laju_aktual


# ==========================================
# EKSEKUSI PROGRAM
# ==========================================
if __name__ == "__main__":

    print("Sedang membaca file Excel...")

    # Path file Excel
    path_file = r"C:\Users\User\OneDrive\Documents\KULIAH\semester 4\METNUM\TUBES\data set india.xlsx"

    # Membaca masing-masing sheet
    df_pm25 = pd.read_excel(path_file, sheet_name='pm25', header=None)
    df_co   = pd.read_excel(path_file, sheet_name='co', header=None)
    df_o3   = pd.read_excel(path_file, sheet_name='o3', header=None)

    # Membuat objek model
    model = ModelKualitasUdara(df_pm25, df_co, df_o3)

    # Menampilkan beberapa data hasil penggabungan
    print("\n--- Cuplikan Data Tergabung ---")
    print(model.df.head())

    # Parameter model
    E_estimasi = 25.0
    k_estimasi = 0.6

    # Simulasi RK4
    hasil_simulasi_rk4 = model.hitung_runge_kutta(
        E=E_estimasi,
        k=k_estimasi
    )

    # Deteksi anomali
    idx_anomali, array_laju = model.deteksi_anomali(
        ambang_batas=30.0
    )

    # Visualisasi
    plt.figure(figsize=(12, 6))

    plt.plot(
        model.waktu,
        model.P_aktual,
        color='blue',
        alpha=0.6,
        label='Data PM2.5 Aktual'
    )

    plt.plot(
        model.waktu,
        hasil_simulasi_rk4,
        color='red',
        linestyle='dashed',
        label=f'Simulasi RK4 (E={E_estimasi}, k={k_estimasi})'
    )

    if len(idx_anomali) > 0:
        plt.scatter(
            model.waktu.iloc[idx_anomali],
            model.P_aktual[idx_anomali],
            color='orange',
            s=50,
            zorder=5,
            label='Titik Anomali'
        )

    plt.title('Simulasi Kualitas Udara & Deteksi Anomali')
    plt.xlabel('Waktu')
    plt.ylabel('Konsentrasi PM2.5 (µg/m³)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()