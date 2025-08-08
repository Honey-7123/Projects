import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Coursera.csv")

df['combined_text'] = (
    df['Course Name'].fillna('') + ' ' +
    df['Course Description'].fillna('') + ' ' +
    df['Skills'].fillna('')
)

tfidf=TfidfVectorizer(stop_words="english")
tfidf_matrix=tfidf.fit_transform(['combined_text'])
indices=pd.Series(df.index,index=df['Course Name']).drop_duplicates()

def recommend_courses(course_name,num_recommendations=5):
    if course_name not in indices:
        return f"course{course_name} not found in the dataset."
    
    idx=indices[course_name]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:num_recommendations+1]
    course_indices=[i[0] for i in sim_scores]

    return df[['Course Name',"University","Course","Rating","Course URL"]]
    sample_course=" creating Dashboard "
    print(recommend_courses(sample_course))
    import joblib
    joblib.dump(df,'courses_df.pkl')
    joblib.dump(tfidf_matrix,'tfidf_matrix.pkl')
    joblib.dump(cosine_sim,"cosine_sim.pkl")
    joblib.dump(indices,'indices.pkl')
    joblib.dump(tfidf,"tfidf_vectorizer.pkl")