import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Model': ['ChatGPT-3.5', 'GPT-4', 'Claude 3', 'Gemini 1 Pro', 'Mixtral-8x7B'],
    'Coherence': [0.88, 0.93, 0.87, 0.84, 0.89],   
    'Fluency': [0.90, 0.94, 0.88, 0.85, 0.91],     
    'ROUGE-1': [0.76, 0.81, 0.74, 0.70, 0.75],   
    'F1 Score': [0.72, 0.79, 0.69, 0.67, 0.73], 
    'Response Time (ms)': [320, 410, 270, 220, 230], 
    'Perplexity': [19, 14, 17, 24, 21], 
    'Parameter Count (B)': [175, 1.8, 0.9, 8, 45] 
}

df = pd.DataFrame(data)

criteria = {
    'Coherence': '+', 
    'Fluency': '+', 
    'ROUGE-1': '+', 
    'F1 Score': '+', 
    'Response Time (ms)': '-', 
    'Perplexity': '-', 
    'Parameter Count (B)': '-'
}

weights = {
    'Coherence': 0.25, 
    'Fluency': 0.20, 
    'ROUGE-1': 0.15, 
    'F1 Score': 0.15, 
    'Response Time (ms)': 0.10, 
    'Perplexity': 0.10, 
    'Parameter Count (B)': 0.05
}

normalized_df = df.copy()
for col in criteria:
    if criteria[col] == '+':
        normalized_df[col] = (df[col] / np.sqrt((df[col]**2).sum())) * weights[col]
    else:
        normalized_df[col] = (1 - (df[col] / np.sqrt((df[col]**2).sum()))) * weights[col]

ideal_best = normalized_df.iloc[:, 1:].max().values
ideal_worst = normalized_df.iloc[:, 1:].min().values

distance_best = np.sqrt(((normalized_df.iloc[:, 1:].values - ideal_best) ** 2).sum(axis=1))
distance_worst = np.sqrt(((normalized_df.iloc[:, 1:].values - ideal_worst) ** 2).sum(axis=1))

topsis_score = distance_worst / (distance_best + distance_worst)

df['TOPSIS Score'] = topsis_score
df['Rank'] = df['TOPSIS Score'].rank(ascending=False)

df.to_csv("topsis_results_conversational.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=df['TOPSIS Score'], y=df['Model'], order=df.sort_values('TOPSIS Score', ascending=False)['Model'])
plt.xlabel("TOPSIS Score")
plt.ylabel("Conversational AI Model")
plt.title("TOPSIS Ranking of Conversational AI Models")
plt.savefig("topsis_ranking_conversational.png")
plt.show()

print("TOPSIS ranking for conversational AI models completed and results saved.")
