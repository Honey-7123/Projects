from fastapi import FastApi,Query
from typing import List,Union
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df=joblib.load("courses_df.pkl")
cosine_sim=joblib.load("cosine_sim.pkl")
indices=joblib.load("indices.pkl")

app=FastApi(title="Api Recommendation")

@app.get("/recommend")
def recommend_courses(course_name:str=Query(...),num_recommendations:int=Query(5)) -> Union[List[dict], str]:
    if course_name not in indices:
        return f"course{course_name} not found in the dataset."
    
    idx=indices[course_name]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:num_recommendations+1]
    course_indices=[i[0] for i in sim_scores]
