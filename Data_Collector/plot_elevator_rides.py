import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_latest_session(sessions_dir="sessions"):
    # Locate the most recently modified directory inside the sessions folder
    if not os.path.exists(sessions_dir):
        print(f"Directory '{sessions_dir}' not found.")
        return

    session_folders = [os.path.join(sessions_dir, d) for d in os.listdir(sessions_dir) 
                       if os.path.isdir(os.path.join(sessions_dir, d))]
    
    if not session_folders:
        print(f"No session folders found in '{sessions_dir}'.")
        return

    latest_session = max(session_folders, key=os.path.getmtime)
    print(f"Plotting latest session: {latest_session}")

    raw_csv_path = os.path.join(latest_session, "raw.csv")
    events_csv_path = os.path.join(latest_session, "events.csv")

    if not os.path.exists(raw_csv_path):
        print(f"No raw.csv found in {latest_session}")
        return

    f_data = []
    s_data = []
    
    with open(raw_csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(',')
            
            # Fast records: F, t_us, ax_g, ay_g, az_g, light_raw, emf_raw, pres_Pa
            if len(parts) >= 8 and parts[1] == 'F':
                try:
                    row = [
                        float(parts[0]), # host_epoch
                        float(parts[3]), # ax_g
                        float(parts[4]), # ay_g
                        float(parts[5]), # az_g
                        float(parts[6]) if parts[6] else np.nan, # light_raw
                        float(parts[7]) if parts[7] else np.nan, # emf_raw
                        float(parts[8]) if len(parts) > 8 and parts[8] else np.nan # pres_Pa
                    ]
                    f_data.append(row)
                except ValueError:
                    pass
                    
            # Slow records: S, t_us, temp_C, rh_pct, gas_kOhm
            elif len(parts) >= 5 and parts[1] == 'S':
                try:
                    row = [
                        float(parts[0]), # host_epoch
                        float(parts[3]), # temp_C
                        float(parts[4]) if len(parts) > 4 and parts[4] else np.nan, # rh_pct
                        float(parts[5]) if len(parts) > 5 and parts[5] else np.nan  # gas_kOhm
                    ]
                    s_data.append(row)
                except ValueError:
                    pass

    if not f_data and not s_data:
        print("No valid sensor data found in raw.csv.")
        return

    df_f = pd.DataFrame(f_data, columns=['host_epoch', 'Accel_X_g', 'Accel_Y_g', 'Accel_Z_g', 'Light_Raw', 'EMF_Raw', 'Pressure_Pa'])
    df_s = pd.DataFrame(s_data, columns=['host_epoch', 'Temp_C', 'RH_pct', 'Gas_kOhm'])

    # Determine a global start time
    start_time = min(df_f['host_epoch'].min() if not df_f.empty else float('inf'),
                     df_s['host_epoch'].min() if not df_s.empty else float('inf'))

    if not df_f.empty: df_f['time_s'] = df_f['host_epoch'] - start_time
    if not df_s.empty: df_s['time_s'] = df_s['host_epoch'] - start_time

    # Setup plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Subplot 1: Acceleration
    if not df_f.empty:
        axes[0].plot(df_f['time_s'], df_f['Accel_X_g'], label='Accel X (g)')
        axes[0].plot(df_f['time_s'], df_f['Accel_Y_g'], label='Accel Y (g)')
        axes[0].plot(df_f['time_s'], df_f['Accel_Z_g'], label='Accel Z (g)')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: Light, EMF, and Pressure
    if not df_f.empty:
        axes[1].plot(df_f['time_s'], df_f['Light_Raw'], label='Light (Raw ADC count)', alpha=0.7)
        axes[1].plot(df_f['time_s'], df_f['EMF_Raw'], label='EMF (Raw ADC count)', alpha=0.7)
        axes[1].set_ylabel('ADC Counts (Light / EMF)')
        axes[1].legend(loc='upper left')
        
        # Pressure is sparse, so we drop NaNs just for this line to plot it connected/with points
        df_pres = df_f[['time_s', 'Pressure_Pa']].dropna()
        if not df_pres.empty:
            ax1b = axes[1].twinx()
            ax1b.plot(df_pres['time_s'], df_pres['Pressure_Pa'], color='purple', marker='.', label='Pressure (Pa)', alpha=0.7)
            ax1b.set_ylabel('Pressure (Pa)', color='purple')
            ax1b.legend(loc='lower right')
            
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Slow Sensors (Temp, Humidity, Gas)
    if not df_s.empty:
        axes[2].plot(df_s['time_s'], df_s['Temp_C'], label='Temperature (°C)', color='red')
        axes[2].set_ylabel('Temperature (°C)', color='red')
        axes[2].legend(loc='upper left')
        
        # Secondary y-axis for Humidity and Gas
        if df_s['RH_pct'].notna().any() or df_s['Gas_kOhm'].notna().any():
            ax2b = axes[2].twinx()
            if df_s['RH_pct'].notna().any():
                ax2b.plot(df_s['time_s'], df_s['RH_pct'], label='Rel. Humidity (%)', color='blue')
            if df_s['Gas_kOhm'].notna().any():
                ax2b.plot(df_s['time_s'], df_s['Gas_kOhm'], label='Gas Resistance (kOhm)', color='green', linestyle='--')
            ax2b.set_ylabel('Humidity (%) / Gas (kOhm)', color='blue')
            ax2b.legend(loc='lower right')

    axes[2].set_xlabel('Time (seconds)')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # Read and overlay events if events.csv is present
    if os.path.exists(events_csv_path):
        df_events = pd.read_csv(events_csv_path)
        
        ride_start_time = None
        max_time = max(df_f['time_s'].max() if not df_f.empty else 0, 
                       df_s['time_s'].max() if not df_s.empty else 0)
                       
        for _, row in df_events.iterrows():
            evt_time = row['host_epoch'] - start_time
            if evt_time < 0:
                continue
                
            event_type = str(row['event_type']).strip()
            
            # Logic to track and draw shaded rides
            if event_type == 'door_close':
                ride_start_time = evt_time
            elif event_type == 'door_open' and ride_start_time is not None:
                for ax in axes:
                    ax.axvspan(ride_start_time, evt_time, color='gray', alpha=0.2)
                ride_start_time = None
            
            # Skip drawing the direction markers entirely
            if event_type == 'direction':
                continue
                
            event_label = f"{row['event_type']} {row['value'] if pd.notna(row['value']) else ''}".strip()
            
            for ax in axes:
                ax.axvline(x=evt_time, color='red', linestyle='--', linewidth=1.2, alpha=0.5)
                # Only put text on the top axis to avoid clutter
                if ax == axes[0]:
                    y_pos = ax.get_ylim()[1] * 0.95
                    ax.text(evt_time + 0.2, y_pos, event_label, rotation=90, color='red', 
                            verticalalignment='top', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                            
        # If the log ended while a ride was still ongoing (doors never reopened)
        if ride_start_time is not None:
            for ax in axes:
                ax.axvspan(ride_start_time, max_time, color='gray', alpha=0.2)

    fig.suptitle(f'Elevator Sensor Telemetry: {os.path.basename(latest_session)}', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_latest_session()
