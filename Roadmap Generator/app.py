#road map Students
import streamlit as st  #interface
import pandas as pd  
import google.generativeai as genai
from datetime import date

#step Env for Genai
genai.configure(api_key="AIzaSyBtXM2MhtYlZZObHvBkMy26ZWhs9WE8mcA")
model=genai.GenerativeModel("gemini-2.0-flash-exp")
#4 parameters
def generate_roadmap(tech_stack,year,speed,goal):
    prompt=f"""
    Generate a Structured day-wise roadmap for a {year} college student learning {tech_stack} 
    learning speed {speed} .Goal: {goal}.
    Format:
    Day | Topic | LeetCode Questions | Youtube Link 
    """
    response = model.generate_content(prompt)
    return response.text

st.set_page_config(page_title="Roadmap Generator", layout="wide")

st.title("🎓 Student Roadmap Generater")

tech_stack=st.selectbox("Chose a Tech Stack:",["python","java","c","c++","Java Script","Data Science"])
year=st.radio("current year",["1 year","2 year","3 year","4 year"])
speed=st.radio("Learning",["Fast","Medium","Slow"])
goal=st.selectbox("End Goal",[
    "master coding Interviews",
    "developer",
    "AI Research"
])
duration_date = st.date_input("Target Completion Date", value=date.today())

#data_range=st.data_input("Duration", [])
#duration_date = st.date_input("Target Completion Date")


if st.button("generate Roadmap"):
    with st.spinner("Working on your roadmap"):
        roadmap_text = generate_roadmap(tech_stack, year, speed, goal)
        roadmap_data=[]
        
        for line in roadmap_text.strip().split("\n"):
            if "|" in line and "Day" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts)>=4:
                    day, topic, lc_q, yt_link = parts[:4]

                    lc_q = f"[LeetCode]({lc_q})" if "http" in lc_q else ("" if lc_q.lower() in ["n/a", "-", ""] else lc_q)

                    yt_link = f"[Watch]({yt_link})" if "http" in yt_link else ("" if yt_link.lower() in ["n/a","-",""] else yt_link) 

                    roadmap_data.append([day, topic, lc_q, yt_link])
#display Table
        if roadmap_data:
            st.subheader("📅 Your Personalized Roadmap")
            formatted_date = duration_date.strftime("%B %d, %Y")
            st.markdown(f"""
            *🎯 Goal*: {goal}  
            *⚡ Speed*: {speed}  
            *📆 Target Completion Date*: {formatted_date}
            """)

            df = pd.DataFrame(roadmap_data, columns=["Day", "Topic", "LeetCode", "YouTube"])
            st.dataframe(df, use_container_width=True)

        else:
            st.warning("❗Couldn't parse the roadmap. Try again.")