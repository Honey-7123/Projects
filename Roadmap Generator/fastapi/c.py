import streamlit as st
import requests

st.title("Recommendation")
st.write("Get Similar Courses from dataset")
num_recs = st.slider("Number of Recommendations", 0, 10, 5)

course_name = st.text_input("Enter the Course name")

if st.button('Get info of rec'):
    if not course_name:
        st.warning("Please enter a course name")
    else:
        url = "http://localhost:8000/recommend"
        params = {'course_name': course_name, 'num_recommendations': num_recs}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, str):
                st.error(data)  # Error message from backend if string
            else:
                st.success("Recommendations:")
                for course in data:
                    st.markdown(f"{course['Course Name']}")
                    st.write(f"University: {course['University']}")
                    st.write(f"Rating: {course['Rating']}")
                    st.markdown("---")
        else:
            st.error("Failed to get recommendations")
