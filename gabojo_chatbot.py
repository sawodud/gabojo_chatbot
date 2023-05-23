import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('tour_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('여행지 추천 챗봇')
st.markdown("[gabojo github주소](https://github.com/sawodud/gabojo)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    top_results = df.nlargest(3, 'distance')

    st.session_state.past.append(user_input)
    st.session_state.generated.append(top_results)

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        generated_data = st.session_state['generated'][i]
        for j, result in enumerate(generated_data.itertuples(), start=1):
            st.write(f"Top {j}")
            st.write("관광지명:", result.관광지명)
            st.write("관광지소개:", result.관광지소개)
            st.write("소재지도로명주소:", result.소재지도로명주소)
            st.write("공공편익시설정보:", result.공공편익시설정보)
            st.write("---")


