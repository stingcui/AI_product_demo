import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval


# Load data
@st.cache_data
def load_data():
    df1 = pd.read_csv('tmdb_5000_credits.csv')
    df2 = pd.read_csv('tmdb_5000_movies.csv')
    df1.columns = ['id', 'tittle', 'cast', 'crew']
    df2 = df2.merge(df1, on='id')
    return df2


# Calculate C and m for weighted rating
def calculate_c_and_m(df):
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.9)
    return C, m


# Calculate weighted rating
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


# Display top movies based on weighted rating
def display_top_movies(df, min_vote_count, selected_genres):
    C, m = calculate_c_and_m(df)
    filtered_df = df.copy()

    if min_vote_count is not None:
        filtered_df = filtered_df[filtered_df['vote_count'] >= min_vote_count]

    if selected_genres:
        filtered_df = filtered_df[
            filtered_df['genres'].apply(lambda x: any(genre['name'] in selected_genres for genre in x))]

    q_movies = filtered_df.loc[filtered_df['vote_count'] >= m]
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1, m=m, C=C)
    q_movies = q_movies.sort_values('score', ascending=False)
    st.header("Top Movies Based on Score")
    st.dataframe(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))


# Display popular movies
def display_popular_movies(df):
    pop = df.sort_values('popularity', ascending=False)
    st.header("Popular Movies")
    plt.figure(figsize=(12, 4))
    plt.barh(pop['title'].head(6), pop['popularity'].head(6), align='center', color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Popularity")
    plt.title("Popular Movies")
    st.pyplot(plt)


# Extract names from the dictionaries in the columns
def extract_names(x):
    if isinstance(x, list):
        return [i['name'] for i in x]
    return []


# Display all movies with pagination, search, and filtering
def display_all_movies(df, min_vote_count, selected_genres, search_query, page):
    filtered_df = df.copy()

    if min_vote_count is not None:
        filtered_df = filtered_df[filtered_df['vote_count'] >= min_vote_count]

    if selected_genres:
        filtered_df = filtered_df[
            filtered_df['genres'].apply(lambda x: any(genre['name'] in selected_genres for genre in x))]

    if search_query:
        filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False)]

    movies_per_page = 50
    total_movies = filtered_df.shape[0]
    total_pages = (total_movies // movies_per_page) + (1 if total_movies % movies_per_page > 0 else 0)

    start_idx = page * movies_per_page
    end_idx = start_idx + movies_per_page
    paginated_df = filtered_df.iloc[start_idx:end_idx]

    # Extract names for the display
    paginated_df['production_companies'] = paginated_df['production_companies'].apply(extract_names)
    paginated_df['production_countries'] = paginated_df['production_countries'].apply(extract_names)
    paginated_df['spoken_languages'] = paginated_df['spoken_languages'].apply(extract_names)

    st.header("All Movies")
    st.dataframe(paginated_df[['title', 'vote_count', 'vote_average', 'popularity', 'production_companies',
                               'production_countries', 'revenue', 'original_language', 'spoken_languages']])

    col1, col2, col3, col4, col5 = st.columns(5)
    if page > 0:
        col1.button("Previous Page", on_click=lambda: st.session_state.update({"page": page - 1}))
    page_numbers = range(1, total_pages + 1)
    selected_page = col2.selectbox("Page", page_numbers, index=page)
    if selected_page != page + 1:
        st.session_state.update({"page": selected_page - 1})
    if page < total_pages - 1:
        col5.button("Next Page", on_click=lambda: st.session_state.update({"page": page + 1}))


# Normalize the titles in the DataFrame
def normalize_titles(df):
    df['normalized_title'] = df['title'].str.lower().str.replace(' ', '')
    return df


# Content-based filtering setup
def content_based_filtering_setup(df):
    df['overview'] = df['overview'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['normalized_title']).drop_duplicates()
    return df, cosine_sim, indices


# Get movie recommendations with normalized titles
def get_recommendations(title, cosine_sim, indices, df):
    normalized_title = title.lower().replace(' ', '')
    if normalized_title in indices:
        idx = indices[normalized_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return df['title'].iloc[movie_indices]
    else:
        return ["No recommendations found."]


# Prepare data for credit, genre, and keyword based recommenders
def prepare_credit_genre_keyword_based(df):
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    df['director'] = df['crew'].apply(get_director)
    for feature in ['cast', 'keywords', 'genres']:
        df[feature] = df[feature].apply(get_list)
    for feature in ['cast', 'keywords', 'director', 'genres']:
        df[feature] = df[feature].apply(clean_data)
    df['soup'] = df.apply(create_soup, axis=1)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    return df, cosine_sim2


# Get the director's name from the crew feature
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Returns the list top 3 elements or entire list
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Create a soup string for each movie
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


# Main function to run the Streamlit app
def main():
    st.title("Movie Recommendation System")

    df = load_data()

    # Ensure genres, production_companies, production_countries, and spoken_languages columns are properly evaluated
    df['genres'] = df['genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['production_companies'] = df['production_companies'].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['production_countries'] = df['production_countries'].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    # Normalize movie titles
    df = normalize_titles(df)

    # Sidebar widgets for filtering
    st.sidebar.header("Filter Options")
    min_vote_count = st.sidebar.slider("Minimum Vote Count", min_value=0, max_value=int(df['vote_count'].max()),
                                       value=0)

    # Extract genre names for multiselect
    all_genres = sorted(list(set(genre['name'] for sublist in df['genres'] for genre in sublist)))
    selected_genres = st.sidebar.multiselect("Select Genres", all_genres)

    # Search movie by name
    search_query = st.sidebar.text_input("Search by movie name or keywords:")

    # Pagination
    if 'page' not in st.session_state:
        st.session_state['page'] = 0

    tab_all_movies, tab_top_movies, tab_popular_movies, tab_recommendations, tab_advanced_recommendations = st.tabs(
        ["All Movies", "Top Movies", "Popular Movies", "Recommendations", "Advanced Recommendations"])

    with tab_all_movies:
        display_all_movies(df, min_vote_count, selected_genres, search_query, st.session_state['page'])

    with tab_top_movies:
        display_top_movies(df, min_vote_count, selected_genres)

    with tab_popular_movies:
        display_popular_movies(df)

    with tab_recommendations:
        st.header("Movie Recommendations")
        df, cosine_sim, indices = content_based_filtering_setup(df)
        movie_title = st.text_input("Enter a movie title for basic recommendations:")
        if movie_title:
            recommendations = get_recommendations(movie_title, cosine_sim, indices, df)
            st.write("Recommended Movies:")
            for movie in recommendations:
                st.write(movie)

    with tab_advanced_recommendations:
        st.header("Advanced Movie Recommendations")
        df, cosine_sim2 = prepare_credit_genre_keyword_based(df)
        movie_title = st.text_input("Enter a movie title for advanced recommendations:")
        if movie_title:
            recommendations = get_recommendations(movie_title, cosine_sim2, indices, df)
            st.write("Recommended Movies:")
            for movie in recommendations:
                st.write(movie)


if __name__ == "__main__":
    main()
