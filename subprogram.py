import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

class ModelKualitasUdara:
    def __init__(self, df_pm25, df_co, df_o3):
        print("Sedang memproses dan menggabungkan dataset...")
        
        df_pm25 = df_pm25.iloc[:, [5, 3]].copy()
        df_pm25.columns = ['Waktu', 'PM25']
        
        df_co = df_co.iloc[:, [5, 3]].copy()
        df_co.columns = ['Waktu', 'CO']
        
        df_o3 = df_o3.iloc[:, [5, 3]].copy()
        df_o3.columns = ['Waktu', 'O3']
        
        df_gabungan = pd.merge(df_pm25, df_co, on='Waktu', how='outer')
        df_gabungan = pd.merge(df_gabungan, df_o3, on='Waktu', how='outer')
        
    
        df_gabungan['Waktu'] = pd.to_datetime(df_gabungan['Waktu'], utc=True)
        df_gabungan = df_gabungan.sort_values('Waktu').reset_index(drop=True)
        

        df_gabungan['PM25'] = pd.to_numeric(df_gabungan['PM25'], errors='coerce').interpolate()
        df_gabungan['CO'] = pd.to_numeric(df_gabungan['CO'], errors='coerce').interpolate()
        df_gabungan['O3'] = pd.to_numeric(df_gabungan['O3'], errors='coerce').interpolate()
        
        self.df = df_gabungan
        self.waktu = self.df['Waktu']
        self.P_aktual = self.df['PM25'].values
        self.CO_aktual = self.df['CO'].values
        self.O3_aktual = self.df['O3'].values
        self.n_data = len(self.P_aktual)
        self.h = 0.25  

    def persamaan_diferensial(self, P, E, k):
        return E - (k * P)

    def hitung_runge_kutta(self, E, k):
        P_simulasi = np.zeros(self.n_data)
        P_simulasi[0] = self.P_aktual[0]
        
        for i in range(self.n_data - 1):
            P_i = P_simulasi[i]
            K1 = self.persamaan_diferensial(P_i, E, k)
            K2 = self.persamaan_diferensial(P_i + (self.h / 2) * K1, E, k)
            K3 = self.persamaan_diferensial(P_i + (self.h / 2) * K2, E, k)
            K4 = self.persamaan_diferensial(P_i + self.h * K3, E, k)
            P_simulasi[i+1] = P_i + (self.h / 6) * (K1 + 2*K2 + 2*K3 + K4)
            
        return P_simulasi

    def hitung_beda_pusat(self, data_array):
        laju = np.zeros(self.n_data)
        for i in range(1, self.n_data - 1):
            laju[i] = (data_array[i+1] - data_array[i-1]) / (2 * self.h)
        laju[0] = (data_array[1] - data_array[0]) / self.h
        laju[-1] = (data_array[-1] - data_array[-2]) / self.h
        return laju

    def deteksi_anomali(self, data_array, ambang_batas, nama_param):
        laju = self.hitung_beda_pusat(data_array)
        idx_anomali = np.where(laju > ambang_batas)[0]
        
        print(f"\n--- DETEKSI ANOMALI {nama_param} (>{ambang_batas}/jam) ---")
        if len(idx_anomali) == 0:
            print(f"Tidak ada anomali {nama_param}")
        else:
            print(f"Ditemukan {len(idx_anomali)} titik anomali {nama_param}")
            for idx in idx_anomali[:5]:
                waktu = self.waktu.iloc[idx].strftime('%Y-%m-%d %H:%M')
                print(f"-> {waktu} | Akselerasi: +{laju[idx]:.2f} | {nama_param}: {data_array[idx]:.1f}")
                
        return idx_anomali



if __name__ == "__main__":


    path_file = r"T:\METNUM\TUBES_METNUM-\data set india.xlsx"

    
    df_pm25 = pd.read_excel(path_file, sheet_name='pm25', header=None)
    df_co   = pd.read_excel(path_file, sheet_name='co', header=None)
    df_o3   = pd.read_excel(path_file, sheet_name='o3', header=None)

    
    model = ModelKualitasUdara(df_pm25, df_co, df_o3)

    
    E_estimasi = 25.0
    k_estimasi = 0.6

    
    hasil_simulasi_rk4 = model.hitung_runge_kutta(E=E_estimasi, k=k_estimasi)

    idx_anomali_pm25 = model.deteksi_anomali(model.P_aktual, ambang_batas=30.0, nama_param="PM2.5")
    idx_anomali_co   = model.deteksi_anomali(model.CO_aktual, ambang_batas=0.5, nama_param="CO")
    idx_anomali_o3   = model.deteksi_anomali(model.O3_aktual, ambang_batas=15.0, nama_param="O3")

    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.plot(model.waktu, model.P_aktual, color='blue', alpha=0.6, linewidth=1.5, label='PM2.5 Aktual')
    ax1.plot(model.waktu, hasil_simulasi_rk4, color='red', linestyle='dashed', linewidth=1.5, 
             label=f'Simulasi RK4 PM2.5 (E={E_estimasi}, k={k_estimasi})')
    ax1.plot(model.waktu, model.O3_aktual, color='green', alpha=0.6, linewidth=1.5, label='O3 Aktual')
    
    if len(idx_anomali_pm25) > 0:
        ax1.scatter(model.waktu.iloc[idx_anomali_pm25], model.P_aktual[idx_anomali_pm25],
                   color='orange', s=50, zorder=5, label='Anomali PM2.5', marker='o')
    if len(idx_anomali_o3) > 0:
        ax1.scatter(model.waktu.iloc[idx_anomali_o3], model.O3_aktual[idx_anomali_o3],
                   color='lime', s=50, zorder=5, label='Anomali O3', marker='s')
    
    ax1.set_xlabel('Waktu', fontsize=12)
    ax1.set_ylabel('PM2.5 & O3 (µg/m³)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(bottom=0)
    
    ax2 = ax1.twinx()
    ax2.plot(model.waktu, model.CO_aktual, color='purple', alpha=0.6, linewidth=1.5, label='CO Aktual')
    
    if len(idx_anomali_co) > 0:
        ax2.scatter(model.waktu.iloc[idx_anomali_co], model.CO_aktual[idx_anomali_co],
                   color='magenta', s=50, zorder=5, label='Anomali CO', marker='^')
    
    ax2.set_ylabel('CO (ppb)', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(bottom=0)
    
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.title('Simulasi Kualitas Udara Multi-Parameter & Deteksi Anomali', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()