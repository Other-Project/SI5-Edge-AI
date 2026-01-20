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

# 3. GÉNÉRATION DES PROFILS INDIVIDUELS
def generate_individual_profiles(config_dict):
    print("--- 1. Génération des profils individuels (Courbes & Moyennes) ---")
    
    for file_key, (time_step, custom_label) in config_dict.items():
        files = [file_key] if isinstance(file_key, str) else file_key
        
        valid_dfs = []
        for f in files:
            if os.path.exists(f):
                valid_dfs.append(pd.read_csv(f))
            else:
                print(f"Saut : {f} non trouvé.")
        
        if not valid_dfs:
            continue

        df = valid_dfs[0].copy()
        col_conso = 'Power' if 'Power' in df.columns else 'Watt (W)'
        
        if len(valid_dfs) > 1:
            for other_df in valid_dfs[1:]:
                df[col_conso] += other_df[col_conso]
        
        models = [p for p in df['Phase'].unique() if p != 'Idle']
        
        plt.figure(figsize=(12, 6))
        averages = {}
        
        idle_data = df[df['Phase'] == 'Idle'][col_conso]
        if not idle_data.empty:
            idle_p = idle_data.mean()
            averages['Idle'] = idle_p

        for model in models:
            indices = df.index[df['Phase'] == model].tolist()
            if not indices: continue
            
            s, e = max(0, indices[0]-5), min(len(df)-1, indices[-1]+5)
            subset = df.iloc[s:e+1].copy()
            subset['Relative Time'] = np.arange(len(subset)) * time_step
            
            label_c = clean_model_name(model)
            plt.plot(subset['Relative Time'], subset[col_conso], label=label_c)
            
            averages[label_c] = df.iloc[indices[0]:indices[-1]+1][col_conso].mean()

        plt.title(f"Profil : {custom_label}")
        plt.xlabel("Secondes")
        plt.ylabel("Puissance (Watt)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        file_name_clean = custom_label.replace(' ', '_').replace('+', 'plus')
        plt.savefig(os.path.join(OUTPUT_DIR, f"courbes_{file_name_clean}.png"))
        plt.close()

        if averages:
            sorted_avg = dict(sorted(averages.items(), key=lambda x: x[1], reverse=True))
            plt.figure(figsize=(10, 5))
            colors = ['orange' if k == 'Idle' else 'skyblue' for k in sorted_avg.keys()]
            bars = plt.bar(sorted_avg.keys(), sorted_avg.values(), color=colors, edgecolor='black')
            
            plt.title(f"Consommation Moyenne : {custom_label}")
            plt.ylabel("Watts")
            plt.xticks(rotation=45, ha='right')
            
            for b in bars:
                plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, 
                         f"{b.get_height():.2f}W", ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"moyennes_{file_name_clean}.png"))
            plt.close()
            
        print(f"Terminé pour {custom_label}")

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
            if not df.empty:
                x_limit = 0.75
                plt.xlim(0, x_limit)

    plt.title(f"Efficacité : {map_label} vs Énergie Dynamique", fontsize=15)
    plt.xlabel(f"Précision ({map_label})", fontsize=12)
    plt.ylabel("Énergie consommée (J)", fontsize=12)
    plt.yscale('log')

    m_leg = [Line2D([0], [0], color=colors[m], lw=2, label=m) for m in models_list]
    leg1 = plt.legend(handles=m_leg, title="Modèles", loc='lower right', fontsize='large', frameon=True, shadow=True)
    plt.gca().add_artist(leg1)

    d_leg = [Line2D([0], [0], marker=markers[d], color='w', label=d, 
                    markerfacecolor='gray', markeredgecolor='black', markersize=9) for d in devices_list]
    plt.legend(handles=d_leg, title="Périphériques", loc='upper left', fontsize='medium', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def generate_global_comparisons(bench_config):
    print("\n--- 2. Génération des comparaisons mAP / Efficacité (Énergie Dynamique) ---")
    results = []

    for cfg in bench_config:
        if not os.path.exists(cfg['bench']):
            print(f"Benchmark non trouvé : {cfg['bench']}")
            continue
            
        dfs_power = [pd.read_csv(pf) for pf in cfg['power_files'] if os.path.exists(pf)]
        if not dfs_power: continue

        idle_p = sum(get_avg_power(df, 'Idle') for df in dfs_power)
        
        df_bench = pd.read_csv(cfg['bench'])
        for _, row in df_bench.iterrows():
            total_model_p = sum(get_avg_power(df, row['Model']) for df in dfs_power)
            
            if total_model_p > 0:
                dynamic_p = max(0, total_model_p - idle_p)
                
                score = (row['time_avg'] * dynamic_p) / 1000.0
                
                results.append({
                    'Device': cfg['device'],
                    'Model': clean_model_name(row['Model']),
                    'mAP_box': row['mAP_box'],
                    'mAP_mask': row['mAP_mask'],
                    'Score': score
                })

    if not results:
        print("Erreur : Aucune donnée trouvée.")
        return

    df_final = pd.DataFrame(results)
    plot_efficiency(df_final, 'mAP_box', 'mAP Box', 'comparaison_efficacite_box.png')
    plot_efficiency(df_final, 'mAP_mask', 'mAP Mask', 'comparaison_efficacite_mask.png')
    print(f"Graphiques globaux sauvegardés dans : {OUTPUT_DIR}")

# 5. EXÉCUTION
if __name__ == "__main__":
    generate_individual_profiles(PROFILING_CONFIG)
    generate_global_comparisons(GLOBAL_BENCH_CONFIG)