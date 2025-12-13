import json
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = "refusal_direction/pipeline/runs/qwen-1_8b-chat/completions"

with open(f"{BASE_PATH}/jailbreakbench_baseline_evaluations.json") as f:
    harmful_baseline = json.load(f)
with open(f"{BASE_PATH}/jailbreakbench_ablation_evaluations.json") as f:
    harmful_ablation = json.load(f)
with open(f"{BASE_PATH}/harmless_baseline_evaluations.json") as f:
    harmless_baseline = json.load(f)
with open(f"{BASE_PATH}/harmless_actadd_evaluations.json") as f:
    harmless_actadd = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
x = np.arange(2)
width = 0.35

baseline_ref = 1 - harmful_baseline["substring_matching_success_rate"]
baseline_safe = 1 - harmful_baseline["llamaguard2_success_rate"]
ablation_ref = 1 - harmful_ablation["substring_matching_success_rate"]
ablation_safe = 1 - harmful_ablation["llamaguard2_success_rate"]

bars1 = ax1.bar(x - width/2, [baseline_ref, baseline_safe], width, label='No intervention', color='steelblue')
bars2 = ax1.bar(x + width/2, [ablation_ref, ablation_safe], width, label='Directional ablation', color='coral', hatch='//')

ax1.set_ylabel('Score')
ax1.set_title('Figure 1: Ablating Refusal Direction\n(Qwen-1.8B-Chat, JailbreakBench)')
ax1.set_xticks(x)
ax1.set_xticklabels(['Refusal Score', 'Safety Score'])
ax1.legend()
ax1.set_ylim(0, 1.1)

for bar in bars1 + bars2:
    h = bar.get_height()
    ax1.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

ax2 = axes[1]
x = np.arange(1)

baseline_ref_harmless = 1 - harmless_baseline["substring_matching_success_rate"]
actadd_ref_harmless = 1 - harmless_actadd["substring_matching_success_rate"]

bars1 = ax2.bar(x - width/2, [baseline_ref_harmless], width, label='No intervention', color='steelblue')
bars2 = ax2.bar(x + width/2, [actadd_ref_harmless], width, label='Activation addition', color='coral', hatch='//')

ax2.set_ylabel('Refusal Score')
ax2.set_title('Figure 3: Adding Refusal Direction\n(Qwen-1.8B-Chat, Harmless Prompts)')
ax2.set_xticks(x)
ax2.set_xticklabels(['Qwen-1.8B'])
ax2.legend()
ax2.set_ylim(0, 1.1)

for bar in bars1 + bars2:
    h = bar.get_height()
    ax2.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('refusal_direction_figures.png', dpi=150)
print("saved refusal_direction_figures.png")
print(f"\nfigure 1 - harmful prompts (JailbreakBench):")
print(f"  baseline: refusal={baseline_ref:.2f}, safety={baseline_safe:.2f}")
print(f"  ablation: refusal={ablation_ref:.2f}, safety={ablation_safe:.2f}")
print(f"\nfigure 3 - harmless prompts:")
print(f"  baseline refusal: {baseline_ref_harmless:.2f}")
print(f"  actadd refusal:   {actadd_ref_harmless:.2f}")