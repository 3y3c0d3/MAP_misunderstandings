import matplotlib.pyplot as plt

folds = [1, 2, 3, 4, 5]
scores = [0.245, 0.238, 0.252, 0.241, 0.249]

plt.figure(figsize=(6,4))
plt.plot(folds, scores, marker='o', linewidth=2)
plt.ylim(0.22, 0.27)
plt.title("Cross-Validation MAP@3 per Fold")
plt.xlabel("Fold")
plt.ylabel("MAP@3")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("assets/cv_scores.png")
plt.show()
