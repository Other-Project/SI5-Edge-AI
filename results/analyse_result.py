import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.lines import Line2D

# 1. CONFIGURATION DES FICHIERS
OUTPUT_DIR = './results/graph_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PROFILING_CONFIG = {
    './results/power_monitoring/rasp_prepared.csv': (1.0, "Raspberry Pi 4"),
    './results/power_monitoring/oakd_prepared.csv': (1.0, "OAK-D Lite"), 
    ('./results/power_monitoring/nvidia_gpu_prepared.csv', './results/power_monitoring/cpu_with_gpu_prepared.csv'): (0.5, "CPU + GPU"),
    './results/power_monitoring/cpu_prepared.csv': (0.5, "CPU Seul"),
}

GLOBAL_BENCH_CONFIG = [
    {'device': 'Raspberry Pi', 'bench': './results/benchmark_rasp.csv', 'power_files': ['./results/power_monitoring/rasp_prepared.csv']},
    {'device': 'OAK-D', 'bench': './results/benchmark_oakd.csv', 'power_files': ['./results/power_monitoring/oakd_prepared.csv']},
    {'device': 'CPU', 'bench': './results/benchmark_cpu.csv', 'power_files': ['./results/power_monitoring/cpu_prepared.csv']},
    {'device': 'CPU + GPU', 'bench': './results/benchmark_gpu.csv', 'power_files': ['./results/power_monitoring/nvidia_gpu_prepared.csv', './results/power_monitoring/cpu_with_gpu_prepared.csv']}
]

# 2. FONCTIONS DE NETTOYAGE ET CALCUL
def clean_model_name(name):
    """Nettoie le nom du modèle pour la légende."""
    name = str(name).replace('.onnx', '').replace('.blob', '')
    if '_openvino' in name:
        name = name.split('_openvino')[0]
    return name

def get_avg_power(df, phase_name):
    """Calcule la puissance moyenne pour une phase spécifique."""
    col = 'Power' if 'Power' in df.columns else 'Watt (W)'
    if 'Phase' not in df.columns: return 0
    subset = df[df['Phase'].astype(str).str.strip().str.lower() == str(phase_name).strip().lower()]
    return subset[col].mean() if not subset.empty else 0

def get_global_max_power(config_dict):
    """Scanne pour trouver la puissance max (Totale)."""
    max_p = 1.0 
    for file_key in config_dict.keys():
        files = [file_key] if isinstance(file_key, str) else file_key
        combined_series = pd.Series(dtype=float)
        for f in files:
            if os.path.exists(f):
                temp_df = pd.read_csv(f)
                col = 'Power' if 'Power' in temp_df.columns else 'Watt (W)'
                combined_series = combined_series.add(temp_df[col], fill_value=0)
        if not combined_series.empty:
            max_p = max(max_p, combined_series.max())
    return max_p

# 3. GÉNÉRATION DES PROFILS INDIVIDUELS
def generate_individual_profiles(config_dict, global_max):
    print(f"--- 1. Génération des profils individuels (Max global: {global_max:.2f}W) ---")
    
    for file_key, (time_step, custom_label) in config_dict.items():
        files = [file_key] if isinstance(file_key, str) else file_key
        df_combined = None
        col_conso = ""
        
        for f in files:
            if not os.path.exists(f): continue
            temp_df = pd.read_csv(f)
            c = 'Power' if 'Power' in temp_df.columns else 'Watt (W)'
            if df_combined is None:
                df_combined = temp_df.copy()
                col_conso = c
            else:
                df_combined[col_conso] = df_combined[col_conso].add(temp_df[c], fill_value=0)
        
        if df_combined is None: continue

        # --- Calcul de l'Idle pour cet appareil ---
        idle_power = get_avg_power(df_combined, 'Idle')
        print(f"   > {custom_label} | Idle détecté : {idle_power:.2f}W")

        models = [p for p in df_combined['Phase'].unique() if p != 'Idle']
        
        # --- Graphique d'évolution (DYNAMIQUE UNIQUEMENT) ---
        plt.figure(figsize=(12, 6))
        averages_total = {}
        
        # On stocke l'idle pour l'histogramme plus tard
        if idle_power > 0:
            averages_total['Idle'] = idle_power

        for model in models:
            indices = df_combined.index[df_combined['Phase'] == model].tolist()
            if not indices: continue
            
            s, e = max(0, indices[0]-5), min(len(df_combined)-1, indices[-1]+5)
            subset = df_combined.iloc[s:e+1].copy()
            subset['Relative Time'] = np.arange(len(subset)) * time_step
            
            # Soustraction de l'idle pour la courbe temporelle
            dynamic_power_curve = (subset[col_conso] - idle_power).clip(lower=0.01)
            
            label_c = clean_model_name(model)
            plt.plot(subset['Relative Time'], dynamic_power_curve, label=label_c)
            
            # On garde la moyenne TOTALE pour l'histogramme
            averages_total[label_c] = df_combined.iloc[indices[0]:indices[-1]+1][col_conso].mean()

        plt.title(f"Puissance Dynamique (Surconsommation) : {custom_label}")
        plt.xlabel("Secondes")
        plt.ylabel("Watts additionnels (Log)")
        #plt.yscale('log')
        plt.ylim(0, (global_max-idle_power)*1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        
        file_name_clean = custom_label.replace(' ', '_').replace('+', 'plus')
        plt.savefig(os.path.join(OUTPUT_DIR, f"courbes_{file_name_clean}.png"))
        plt.close()

        # --- Graphique des moyennes (TOTALES AVEC IDLE) ---
        if averages_total:
            # Tri décroissant par puissance
            sorted_avg = dict(sorted(averages_total.items(), key=lambda x: x[1], reverse=True))
            plt.figure(figsize=(10, 5))
            
            # Couleur différente pour l'Idle
            colors = ['#ff9933' if k == 'Idle' else '#66b3ff' for k in sorted_avg.keys()]
            bars = plt.bar(sorted_avg.keys(), sorted_avg.values(), color=colors, edgecolor='black')
            
            plt.title(f"Consommation Moyenne Totale : {custom_label} (avec Idle)")
            plt.ylabel("Watts (Log)")
            plt.yscale('log')
            plt.ylim(1, global_max * 1.2 )
            plt.xticks(rotation=45, ha='right')
            
            for b in bars:
                val = b.get_height()
                if val >= 1:
                    plt.text(b.get_x()+b.get_width()/2, val * 1.05, 
                             f"{val:.2f}W", ha='center', fontweight='bold', fontsize=9)
            
            plt.grid(True, which="both", axis='y', ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"moyennes_{file_name_clean}.png"))
            plt.close()
            
        print(f"   > Graphiques terminés pour {custom_label}")

# 4. GÉNÉRATION DES COMPARAISONS GLOBALES
def plot_efficiency(df, map_col, map_label, filename):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    models_list = sorted(df['Model'].unique())
    devices_list = df['Device'].unique()
    colors = dict(zip(models_list, sns.color_palette("husl", len(models_list))))
    markers = {dev: m for dev, m in zip(devices_list, ['o', 's', 'D', '^', 'v'])}

    for mod in models_list:
        mod_df = df[df['Model'] == mod].sort_values(map_col)
        plt.plot(mod_df[map_col], mod_df['Score'], color=colors[mod], alpha=0.4, linewidth=1.5)
        for dev in devices_list:
            pt = mod_df[mod_df['Device'] == dev]
            if not pt.empty:
                plt.scatter(pt[map_col], pt['Score'], color=colors[mod], marker=markers[dev], 
                            s=150, edgecolors='black', alpha=0.9, zorder=5)

    plt.title(f"Efficacité Énergétique Dynamique : {map_label} vs Joules", fontsize=15)
    plt.xlabel(f"Précision ({map_label})", fontsize=12)
    plt.ylabel("Énergie Dynamique Totale (Joules)")
    plt.yscale('log')

    # Légendes
    m_leg = [Line2D([0], [0], color=colors[m], lw=2, label=m) for m in models_list]
    leg1 = plt.legend(handles=m_leg, title="Modèles", loc='lower right', frameon=True, shadow=True)
    plt.gca().add_artist(leg1)
    d_leg = [Line2D([0], [0], marker=markers[d], color='w', label=d, markerfacecolor='gray', markersize=9) for d in devices_list]
    plt.legend(handles=d_leg, title="Périphériques", loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def generate_global_comparisons(bench_config):
    print("\n--- 2. Génération des comparaisons mAP / Efficacité ---")
    results = []
    for cfg in bench_config:
        if not os.path.exists(cfg['bench']): continue
        dfs_power = [pd.read_csv(pf) for pf in cfg['power_files'] if os.path.exists(pf)]
        if not dfs_power: continue

        idle_p = sum(get_avg_power(df, 'Idle') for df in dfs_power)
        df_bench = pd.read_csv(cfg['bench'])

        for _, row in df_bench.iterrows():
            total_model_p = sum(get_avg_power(df, row['Model']) for df in dfs_power)
            if total_model_p > 0:
                dynamic_p = max(0.01, total_model_p - idle_p)
                score = (row['time_avg'] / 1000.0) * dynamic_p
                results.append({
                    'Device': cfg['device'],
                    'Model': clean_model_name(row['Model']),
                    'mAP_box': row['mAP_box'],
                    'mAP_mask': row['mAP_mask'],
                    'Score': score
                })

    if not results: return
    df_final = pd.DataFrame(results)
    plot_efficiency(df_final, 'mAP_box', 'mAP Box', 'comparaison_efficacite_box.png')
    plot_efficiency(df_final, 'mAP_mask', 'mAP Mask', 'comparaison_efficacite_mask.png')

# 5. EXÉCUTION
if __name__ == "__main__":
    # Max global pour harmoniser tous les axes Y
    global_max_found = get_global_max_power(PROFILING_CONFIG)
    
    generate_individual_profiles(PROFILING_CONFIG, global_max_found)
    generate_global_comparisons(GLOBAL_BENCH_CONFIG)
    
    print(f"\nTraitement terminé. Les graphes temporels sont DYNAMIQUES, les moyennes sont TOTALES (avec Idle).")