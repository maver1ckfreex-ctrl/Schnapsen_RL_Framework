import pandas as pd
import numpy as np
from scipy.stats import norm

def process_win_rate_matrix(input_file, output_file, n_games=10000, alpha=0.05, use_p_value=True):
    """
    Processes the win rate matrix and performs a Z-test.
    
    Parameters:
    - input_file: Path to the CSV winning rate matrix.
    - output_file: Path where the result will be saved.
    - n_games: Number of games played for each cell (default 10,000).
    - alpha: Significance level (default 0.05 for 95% confidence).
    - use_p_value: If True, outputs p-values. If False, outputs Z-scores.
    """
    
    # Load the data
    df = pd.read_csv(input_file, index_col=0)
    
    # Standard Error for the null hypothesis (p=0.5)
    # SE = sqrt( p0 * (1 - p0) / n )
    se = np.sqrt(0.5 * 0.5 / n_games)
    
    def calculate_cell(win_rate):
        # Calculate Z-score
        z_score = (win_rate - 0.5) / se
        
        # Calculate Two-tailed p-value
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        # Determine conclusion
        if p_value < alpha:
            conclusion = "B" if win_rate > 0.5 else "W"
        else:
            conclusion = "E"
            
        # Select value for output
        val = p_value if use_p_value else z_score
        
        return f"{conclusion} | {val:.4f}"

    # Apply the function to all numeric columns
    result_df = df.applymap(calculate_cell)
    
    # Save the result
    result_df.to_csv(output_file)
    print(f"Processed matrix saved to: {output_file}")
    return result_df

# --- SETTINGS --- choose which matrix you want use to generate z-test result
INPUT_FILENAME = 'winning_rate_matrix_10k_base.csv'
#INPUT_FILENAME = 'winning_rate_matrix_10k_SL.csv'
#INPUT_FILENAME = 'winning_rate_matrix_10k_RL.csv'
OUTPUT_FILENAME = 'z_test_base.csv'
#OUTPUT_FILENAME = 'z_test_SL.csv'
#OUTPUT_FILENAME = 'z_test_RL.csv'
GAMES_PER_MATCH = 10000
SIGNIFICANCE_LEVEL = 0.05
SHOW_P_VALUE = True  # Set to False to show Z-scores instead

# Execute
if __name__ == "__main__":
    final_matrix = process_win_rate_matrix(
        input_file=INPUT_FILENAME,
        output_file=OUTPUT_FILENAME,
        n_games=GAMES_PER_MATCH,
        alpha=SIGNIFICANCE_LEVEL,
        use_p_value=SHOW_P_VALUE
    )
    
    # Display the first few rows of the result
    print("\nPreview of the new matrix:")
    print(final_matrix.head())
