import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None


def extract_movie_minutes(duration_value):
    if isinstance(duration_value, str) and duration_value.endswith(" min"):
        number_text = duration_value.split()[0]
        if number_text.isdigit():
            return int(number_text)
    return pd.NA


def load_netflix_data(csv_file="netflix_titles.csv"):
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {csv_file}")


def clean_data(df):
    cleaned = df.copy()
    cleaned["country"] = cleaned["country"].fillna("Unknown")
    cleaned["director"] = cleaned["director"].fillna("Unknown")
    cleaned["rating"] = cleaned["rating"].fillna("Unknown")
    cleaned["duration"] = cleaned["duration"].fillna("Unknown")
    cleaned["release_year"] = pd.to_numeric(cleaned["release_year"], errors="coerce")
    cleaned["movie_minutes"] = cleaned["duration"].map(extract_movie_minutes)
    return cleaned


def explode_genres(df):
    genre_data = df["listed_in"].fillna("").str.split(",", expand=False)
    genres = []
    for row in genre_data:
        if not isinstance(row, list):
            continue
        for item in row:
            genres.append(item.strip())
    return pd.Series(genres, name="genre")


def explode_countries(df):
    country_data = df["country"].fillna("").str.split(",", expand=False)
    countries = []
    for row in country_data:
        if not isinstance(row, list):
            continue
        for item in row:
            countries.append(item.strip())
    return pd.Series(countries, name="country_name")


def age_group_mapping(rating):
    if rating in ["G", "TV-Y", "TV-G"]:
        return "All Ages"
    if rating in ["TV-Y7", "TV-Y7-FV", "PG"]:
        return "Children"
    if rating in ["PG-13", "TV-14"]:
        return "Teens"
    if rating in ["R", "TV-MA", "NC-17"]:
        return "Adults"
    return "Other/Unknown"


def top_genres(df, n=10):
    all_genres = explode_genres(df)
    return all_genres.value_counts().head(n)


def country_distribution(df, n=10):
    all_countries = explode_countries(df)
    country_counts = all_countries.value_counts().head(n)
    return country_counts


def rating_age_group_distribution(df):
    rating_df = df[["rating"]].copy()
    rating_df["age_group"] = rating_df["rating"].map(age_group_mapping)
    return rating_df["age_group"].value_counts()


def predict_titles_by_country_and_director(df, country=None, director=None, top_n=5):
    if country is None and director is None:
        return pd.DataFrame(columns=["title", "director", "country", "rating", "listed_in"])

    filtered = df.copy()
    if country:
        country_value = country.strip().lower()
        filtered = filtered[filtered["country"].fillna("").str.lower().str.contains(country_value, na=False)]
    if director:
        director_value = director.strip().lower()
        filtered = filtered[filtered["director"].fillna("").str.lower().str.contains(director_value, na=False)]

    if filtered.empty:
        fallback = df.copy()
        if country:
            fallback = fallback[fallback["country"].fillna("").str.lower().str.contains(country.strip().lower(), na=False)]
        if director:
            fallback = fallback[fallback["director"].fillna("").str.lower().str.contains(director.strip().lower(), na=False)]
        filtered = fallback

    if filtered.empty:
        return pd.DataFrame(columns=["title", "director", "country", "rating", "listed_in"])

    filtered = filtered.sort_values(["release_year", "title"], ascending=[False, True])
    result = filtered[["title", "director", "country", "rating", "listed_in"]].drop_duplicates().head(top_n)
    return result.reset_index(drop=True)


def plot_genre_popularity(df):
    genre_counts = top_genres(df, 10)
    data = pd.DataFrame({"genre": genre_counts.index, "count": genre_counts.values})
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="count", y="genre", palette="viridis", dodge=False)
    plt.title("Most Popular Genres on Netflix")
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig("popular_genres.png", dpi=200)
    plt.close()


def plot_country_distribution(df):
    country_counts = country_distribution(df, 10)
    data = pd.DataFrame({"country": country_counts.index, "count": country_counts.values})
    plt.figure(figsize=(12, 7))
    sns.barplot(data=data, x="count", y="country", palette="magma", dodge=False)
    plt.title("Country-wise Netflix Content Distribution")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig("country_distribution.png", dpi=200)
    plt.close()


def plot_rating_age_groups(df):
    rating_counts = rating_age_group_distribution(df)
    data = pd.DataFrame({"age_group": rating_counts.index, "count": rating_counts.values})
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x="count", y="age_group", palette="Set2", dodge=False)
    plt.title("Netflix Content Ratings by Age Group")
    plt.xlabel("Number of Titles")
    plt.ylabel("Age Group")
    plt.tight_layout()
    plt.savefig("rating_by_age_group.png", dpi=200)
    plt.close()


def plot_year_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="release_year", kde=True, bins=30, color="steelblue")
    plt.title("Distribution of Release Years")
    plt.xlabel("Release Year")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("release_year_distribution.png", dpi=200)
    plt.close()


def plot_type_distribution(df):
    type_counts = df["type"].value_counts()
    data = pd.DataFrame({"type": df["type"]})
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="type", palette="viridis")
    plt.title("Count of Movies vs TV Shows")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("type_distribution.png", dpi=200)
    plt.close()


def plot_runtime_boxplot(df):
    valid_runtime = df[df["movie_minutes"].notna()]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=valid_runtime, x="type", y="movie_minutes", palette="pastel")
    plt.title("Runtime Comparison by Type")
    plt.xlabel("Type")
    plt.ylabel("Duration (Minutes)")
    plt.tight_layout()
    plt.savefig("runtime_boxplot.png", dpi=200)
    plt.close()


def print_menu():
    print("\n====================================")
    print("NETFLIX ANALYSIS MENU")
    print("====================================")
    print("1. Show top genres")
    print("2. Show country-wise distribution")
    print("3. Show ratings by age group")
    print("4. Show type counts (Movies vs TV Shows)")
    print("5. Show release year distribution")
    print("6. Make top genres chart")
    print("7. Make country distribution chart")
    print("8. Make age-group rating chart")
    print("9. Make runtime boxplot")
    print("10. Predict titles by country/director")
    print("11. Search titles for exact information")
    print("12. Show dataset summary")
    print("13. Exit")


def search_title_info(df, title_keyword):
    matches = df[df["title"].str.contains(title_keyword, case=False, na=False)]
    if matches.empty:
        print("No titles matched your search.")
        return

    result = matches[["title", "type", "director", "country", "rating", "release_year", "listed_in"]].head(10)
    print(result.to_string(index=False))


def main():
    try:
        df = load_netflix_data("netflix_titles.csv")
    except FileNotFoundError as exc:
        print(str(exc))
        return

    df = clean_data(df)
    sns.set_theme(style="whitegrid")

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            print("\nTop 10 genres:")
            print(top_genres(df, 10))
        elif choice == "2":
            print("\nTop 10 countries by content count:")
            print(country_distribution(df, 10))
        elif choice == "3":
            print("\nAge group ratings:")
            print(rating_age_group_distribution(df))
        elif choice == "4":
            print("\nType counts:\n", df["type"].value_counts())
        elif choice == "5":
            plot_year_distribution(df)
        elif choice == "6":
            plot_genre_popularity(df)
        elif choice == "7":
            plot_country_distribution(df)
        elif choice == "8":
            plot_rating_age_groups(df)
        elif choice == "9":
            plot_runtime_boxplot(df)
        elif choice == "10":
            country = input("Enter a country name (or press Enter for all): ").strip() or None
            director = input("Enter a director name (or press Enter for all): ").strip() or None
            predicted = predict_titles_by_country_and_director(df, country=country, director=director, top_n=5)
            print("\nPredicted titles based on country/director:")
            if predicted.empty:
                print("No matching titles found.")
            else:
                print(predicted.to_string(index=False))
        elif choice == "11":
            keyword = input("Enter a title keyword to search: ").strip()
            if not keyword:
                print("Please enter a keyword.")
                continue
            search_title_info(df, keyword)
        elif choice == "12":
            print("\nDataset preview:")
            print(df.head())
            print("\nDataset shape:", df.shape)
            print("\nMissing values:\n", df.isna().sum())
            print("\nAverage movie duration:", df["movie_minutes"].mean())
        elif choice == "13":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()