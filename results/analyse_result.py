import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.lines import Line2D

# 1. CONFIGURATION DES FICHIERS
OUTPUT_DIR = './results/graph_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROFILING_CONFIG = {
    './results/power_monitoring/rasp_prepared.csv': (1.0, "Raspberry Pi 4"),
    './results/power_monitoring/oakd_prepared.csv': (1.0, "OAK-D Lite"), 
    ('./results/power_monitoring/nvidia_gpu_prepared.csv', './results/power_monitoring/cpu_with_gpu_prepared.csv'): (0.5, "GPU"),
    './results/power_monitoring/cpu_prepared.csv': (0.5, "CPU Seul"),
}

GLOBAL_BENCH_CONFIG = [
    {'device': 'Raspberry Pi', 'bench': './results/benchmark_rasp.csv', 'power_files': ['./results/power_monitoring/rasp_prepared.csv']},
    {'device': 'OAK-D', 'bench': './results/benchmark_oakd.csv', 'power_files': ['./results/power_monitoring/oakd_prepared.csv']},
    {'device': 'CPU', 'bench': './results/benchmark_cpu.csv', 'power_files': ['./results/power_monitoring/cpu_prepared.csv']},
    {'device': 'GPU', 'bench': './results/benchmark_gpu.csv', 'power_files': ['./results/power_monitoring/nvidia_gpu_prepared.csv', './results/power_monitoring/cpu_with_gpu_prepared.csv']}
]

# 2. FONCTIONS DE NETTOYAGE ET CALCUL
def clean_model_name(name):
    name = str(name).replace('.onnx', '').replace('.blob', '')
    if '_openvino' in name:
        name = name.split('_openvino')[0]
    return name

def get_avg_power(df, phase_name):
    col = 'Power' if 'Power' in df.columns else 'Watt (W)'
    col_phase = 'Phase' if 'Phase' in df.columns else 'Model'
    if col_phase not in df.columns: return 0
    subset = df[df[col_phase].astype(str).str.strip().str.lower() == str(phase_name).strip().lower()]
    return subset[col].mean() if not subset.empty else 0

def get_global_max_powers(config_dict):
    max_tot = 0
    max_dyn = 0
    for file_key in config_dict.keys():
        files = [file_key] if isinstance(file_key, str) else list(file_key)
        dfs = [pd.read_csv(f) for f in files if os.path.exists(f)]
        if not dfs: continue
        col = 'Power' if 'Power' in dfs[0].columns else 'Watt (W)'
        min_len = min(len(d) for d in dfs)
        combined = np.zeros(min_len)
        for d in dfs:
            combined += d[col].iloc[:min_len].values
        
        df_tmp = dfs[0].iloc[:min_len].copy()
        df_tmp[col] = combined
        idle = get_avg_power(df_tmp, 'Idle')
        
        max_tot = max(max_tot, np.max(combined))
        max_dyn = max(max_dyn, np.max(combined - idle))
    return max_tot, max_dyn

# 3. GÉNÉRATION DES PROFILS INDIVIDUELS
def generate_individual_profiles(config_dict, g_max_total, g_max_dynamic):
    print(f"--- 1. Profils individuels (Axes Y verrouillés) ---")
    
    for file_key, (time_step, custom_label) in config_dict.items():
        files = [file_key] if isinstance(file_key, str) else list(file_key)
        valid_dfs = [pd.read_csv(f) for f in files if os.path.exists(f)]
        if not valid_dfs: continue

        col_conso = 'Power' if 'Power' in valid_dfs[0].columns else 'Watt (W)'
        col_phase = 'Phase' if 'Phase' in valid_dfs[0].columns else 'Model'
        
        min_len = min(len(d) for d in valid_dfs)
        df_total = valid_dfs[0].iloc[:min_len].copy()
        if len(valid_dfs) > 1:
            for other_df in valid_dfs[1:]:
                df_total[col_conso] += other_df[col_conso].iloc[:min_len].values

        idle_power = get_avg_power(df_total, 'Idle')
        models = [p for p in df_total[col_phase].unique() if p.lower() != 'idle']
        
        # --- A. COURBES DYNAMIQUES ---
        plt.figure(figsize=(12, 6))
        for model in models:
            indices = df_total.index[df_total[col_phase] == model].tolist()
            if not indices: continue
            s, e = max(0, indices[0]-5), min(len(df_total)-1, indices[-1]+5)
            subset = df_total.iloc[s:e+1].copy()
            subset['Time'] = np.arange(len(subset)) * time_step
            dyn_vals = (subset[col_conso] - idle_power).clip(lower=0) 
            plt.plot(subset['Time'], dyn_vals, label=clean_model_name(model))

        plt.title(f"Évolution de la puissance en fonction des modèles : {custom_label}")
        plt.ylabel("Watts")
        plt.xlabel("Secondes")
        plt.ylim(0, g_max_dynamic * 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"courbes_{custom_label.replace(' ', '_')}.png"))
        plt.close()

        # --- B. MOYENNES TOTALES (Logarithmique) ---
        phases = ['Idle'] + models
        avg_data = {clean_model_name(p) if p.lower() != 'idle' else 'Idle': 
                    [get_avg_power(d, p) for d in valid_dfs] for p in phases}
        sorted_keys = sorted(avg_data.keys(), key=lambda k: sum(avg_data[k]), reverse=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bottoms = np.zeros(len(sorted_keys))
        colors = sns.color_palette("muted", len(valid_dfs))
        
        for i in range(len(valid_dfs)):
            heights = [avg_data[k][i] for k in sorted_keys]
            label = "GPU" if (i==0 and "GPU" in custom_label) else ("CPU" if "GPU" in custom_label else f"File {i+1}")
            ax.bar(sorted_keys, heights, bottom=bottoms, color=colors[i], edgecolor='black', label=label)
            bottoms += np.array(heights)

        for idx, k in enumerate(sorted_keys):
            total = sum(avg_data[k])
            ax.text(idx, total * 1.05, f"{total:.2f}W", ha='center', fontweight='bold')

        ax.set_title(f"Consommation Moyenne : {custom_label}")
        ax.set_ylabel("Watts") # Label normal
        ax.set_yscale('log')
        ax.set_ylim(bottom=1, top=g_max_total * 2)
        ax.grid(True, which="both", linestyle='--', alpha=0.4)
        
        # Ajout du texte "Logarithmique" en haut de l'axe Y
        ax.text(0, 1.02, 'Logarithmique', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

        plt.xticks(rotation=45, ha='right')
        if len(valid_dfs) > 1:
            ax.legend(title="Composants")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"moyennes_totales_{custom_label.replace(' ', '_')}.png"))
        plt.close()

# 4. COMPARAISONS GLOBALES
def plot_efficiency(df, map_col, map_label, filename):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    models_list = sorted(df['Model'].unique())
    devices_list = df['Device'].unique()
    colors = dict(zip(models_list, sns.color_palette("husl", len(models_list))))
    markers = {dev: m for dev, m in zip(devices_list, ['o', 's', 'D', '^', 'v'])}

    for mod in models_list:
        mod_df = df[df['Model'] == mod].sort_values(map_col)
        ax.plot(mod_df[map_col], mod_df['Score'], color=colors[mod], alpha=0.4, linewidth=1.5)
        for dev in devices_list:
            pt = mod_df[mod_df['Device'] == dev]
            if not pt.empty:
                ax.scatter(pt[map_col], pt['Score'], color=colors[mod], marker=markers[dev], 
                            s=150, edgecolors='black', alpha=0.9, zorder=5)

    ax.set_xlim(0, 0.70)
    ax.set_title(f"Efficacité : {map_label} vs Énergie Dynamique", fontsize=15)
    ax.set_xlabel(f"Précision ({map_label})", fontsize=12)
    ax.set_ylabel("Énergie consommée (J)", fontsize=12)
    
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    # Texte Logarithmique en haut
    ax.text(0, 1.01, 'Logarithmique', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')

    m_leg = [Line2D([0], [0], color=colors[m], lw=2, label=m) for m in models_list]
    leg1 = ax.legend(handles=m_leg, title="Modèles", loc='lower right', fontsize='large', frameon=True, shadow=True)
    ax.add_artist(leg1)

    d_leg = [Line2D([0], [0], marker=markers[d], color='w', label=d, 
                    markerfacecolor='gray', markeredgecolor='black', markersize=9) for d in devices_list]
    ax.legend(handles=d_leg, title="Périphériques", loc='upper left', fontsize='medium', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_inference_time_bar(df, filename):
    pivot_df = df.pivot(index='Model', columns='Device', values='InferenceTime')
    ax = pivot_df.plot(kind='bar', edgecolor='black', figsize=(12,6))
    
    plt.title("Temps d'inférence moyen par modèle et périphérique", fontsize=15)
    plt.xlabel("Modèle", fontsize=12)
    plt.ylabel("Temps d'inférence (ms)", fontsize=12)
    
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    # Texte Logarithmique en haut
    plt.text(0, 1.01, 'Logarithmique', transform=plt.gca().transAxes, fontsize=9, fontweight='bold', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Périphérique", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def generate_global_comparisons(bench_config):
    print("--- 2. Comparaisons Globales (mAP, Joules, Temps) ---")
    results = []
    for cfg in bench_config:
        if not os.path.exists(cfg['bench']): continue
        dfs_p = [pd.read_csv(pf) for pf in cfg['power_files'] if os.path.exists(pf)]
        if not dfs_p: continue
        idle_p = sum(get_avg_power(df, 'Idle') for df in dfs_p)
        df_bench = pd.read_csv(cfg['bench'])

        for _, row in df_bench.iterrows():
            total_p = sum(get_avg_power(df, row['Model']) for df in dfs_p)
            if total_p > 0:
                dyn_p = max(0.01, total_p - idle_p)
                score_joules = (row['time_avg'] / 1000.0) * dyn_p
                results.append({
                    'Device': cfg['device'], 'Model': clean_model_name(row['Model']),
                    'mAP_box': row.get('mAP_box', 0), 'mAP_mask': row.get('mAP_mask', 0),
                    'Score': score_joules, 'InferenceTime': row['time_avg']
                })

    if results:
        df_res = pd.DataFrame(results)
        plot_efficiency(df_res, 'mAP_box', 'mAP Box', 'comparaison_efficacite_box.png')
        plot_efficiency(df_res, 'mAP_mask', 'mAP Mask', 'comparaison_efficacite_mask.png')
        plot_inference_time_bar(df_res, 'comparaison_temps_inference.png')

# 5. EXECUTION
if __name__ == "__main__":
    max_tot, max_dyn = get_global_max_powers(PROFILING_CONFIG)
    generate_individual_profiles(PROFILING_CONFIG, max_tot, max_dyn)
    generate_global_comparisons(GLOBAL_BENCH_CONFIG)
    print(f"\nTraitement terminé. Résultats dans {OUTPUT_DIR}")