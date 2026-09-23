"""Generate vocab_growth.png plot from ablation_results.json."""
import json
import matplotlib.pyplot as plt

with open('ablation_results.json', encoding='utf-8') as f:
    data = json.load(f)

growth = data['vocabulary_growth']
xs = [r['rows'] for r in growth]
ys = [r['templates'] for r in growth]

plt.figure(figsize=(5.5, 3.0))
plt.plot(xs, ys, marker='o', linewidth=1.8, color='#1f77b4')
for x, y in zip(xs, ys):
    plt.annotate(str(y), xy=(x, y), xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=8, color='#333333')
plt.xlabel('Số dòng log đã parse')
plt.ylabel('Số event template')
plt.title('Vocabulary growth của Drain3 trên HDFS')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()
plt.savefig('images/vocab_growth.png', dpi=150, bbox_inches='tight')
print('Saved images/vocab_growth.png')
