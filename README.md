# Conversational AI Model Ranking using TOPSIS

This project evaluates and ranks various conversational AI models using the **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) method. The models assessed include:

- ChatGPT-3.5
- GPT-4
- Claude 3
- Gemini 1 Pro
- Mixtral-8x7B

These models are evaluated based on multiple performance criteria such as:

- **Coherence**
- **Fluency**
- **ROUGE-1**
- **F1 Score**
- **Response Time**
- **Perplexity**
- **Parameter Count**

The performance scores are normalized and weighted according to their importance, and the models are ranked based on their proximity to the ideal solution. The ranking results are visualized in a bar plot and saved to a CSV file for further analysis.

## Features

- Evaluates models based on various performance metrics.
- Normalizes and weights the criteria using custom weights.
- Ranks models using the **TOPSIS** method.
- Saves ranking results and visualizations for further analysis.

## Files

- **topsis_results_conversational.csv**: Stores the ranking results.
- **topsis_ranking_conversational.png**: Bar plot of the ranking.

## How to Run

1. Install the required dependencies:
    ```bash
    pip install numpy pandas matplotlib seaborn
    ```
2. Run the script to generate rankings and visualizations.

3. View the output files:
    - **topsis_results_conversational.csv** for model ranking data.
    - **topsis_ranking_conversational.png** for a graphical representation.
