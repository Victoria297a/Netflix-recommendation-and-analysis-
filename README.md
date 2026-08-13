# Netflix Content Analysis

This project analyzes the Netflix titles dataset and explores:

- Most popular genres
- Country-wise content distribution
- Content ratings by age group
- Movie runtime patterns
- Release-year trends
- Prediction examples using country and director filters

## Files

- `movies.py` — main analysis and plotting script
- `netflix_titles.csv` — dataset used for analysis
- Generated plots:
  - `popular_genres.png`
  - `country_distribution.png`
  - `rating_by_age_group.png`
  - `release_year_distribution.png`
  - `type_distribution.png`
  - `runtime_boxplot.png`

## How to run

1. Open a terminal in this folder.
2. Activate the virtual environment if needed.
3. Run:

```bash
python movies.py
```

## Notes

- The script is currently commented out at the bottom so it can be imported safely in other scripts.
- To run it manually, uncomment the `if __name__ == "__main__":` block in `movies.py`.
- The dataset is expected to be named `netflix_titles.csv` and located in the same folder.

## Example analysis included

The script performs:

- Genre frequency counts
- Country-based counts
- Age-group classification from ratings
- Average movie duration calculation
- Visual plot generation with seaborn and matplotlib
- Prediction-style filtering by country and director
