import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import json
def refresh_chat():
    # Clear the chat session state
    st.session_state['generated'] = []
    st.session_state['past'] = []

    # Rerun the Streamlit app by modifying a non-existent widget
    st.sidebar.empty()

def show_home():
    st.title("Gabojo Chat")
    st.subheader("Gabojo Chat??")
    st.write("가보조 챗은 사용자들의 취향을 고려해서 여행지를 추천해주는 맞춤형 챗봇입니다.\n")
    st.subheader("사용법")
    st.write("메뉴에 챗봇을 눌러 원하는 여행 지역, 취향을 적고 일정에 추가해주세여~")

def show_travel_chatbot():
    @st.cache(allow_output_mutation=True)
    def cached_model():
        model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        return model

    @st.cache(allow_output_mutation=True)
    def get_dataset():
        df = pd.read_csv('tour_dataset.csv')
        df['embedding'] = df['embedding'].apply(json.loads)
        return df
    
    @st.cache(allow_output_mutation=True)
    def get_course_dataset():
        df = pd.read_csv('tour_course.csv')
        df['embedding'] = df['embedding'].apply(json.loads)
        return df    

    model = cached_model()
    df = get_dataset()
    course_df = get_course_dataset()

    st.header('여행지 추천 챗봇')
    st.markdown("[gabojo github주소](https://github.com/sawodud/gabojo)")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []
        
    if st.button('Refresh Chat'):
        refresh_chat()
        
    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('당신: ', '')
        submitted = st.form_submit_button('전송')

    if submitted and user_input:
        if user_input == '다른여행지' and st.session_state.past:
            previous_input = st.session_state.past[-1]
            user_input = previous_input

        embedding = model.encode(user_input)

        df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        course_df['distance'] = course_df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        top_3_indices = df['distance'].nlargest(3).index
        top_course_indices = course_df['distance'].nlargest(5).index

        if '코스' in user_input and '여행' in user_input:
            if course_df.loc[top_course_indices[0], 'distance'] >= 0.7:
                answers = course_df.loc[top_course_indices]
                for idx, answer in answers.iterrows():
                    st.session_state.generated.append((user_input, "에 대한 유사한 여행 코스 추천입니다.\n", "여행 코스명: ", answer['시티투어코스명'], "\n", "코스 정보: ", answer['시티투어코스정보'], "\n"))
            else:
                st.session_state.generated.append((user_input, "무슨말인지 모르겠습니다. 더 구체적인 정보를 입력해주세요.\n ex) '지역이름' '즐기고싶은 것' 등을 적어주세요."))
        else:
            if df.loc[top_3_indices[0], 'distance'] >= 0.6:
                answers = df.loc[top_3_indices]

                for idx, answer in answers.iterrows():
                    st.session_state.generated.append((user_input, "에 대한 추천 여행지입니다.\n", "관광지명: ", answer['관광지명'], "\n", "여행지 정보: ", answer['관광지소개'], "\n주소: ", answer['소재지도로명주소'], "\n 공공편익시설정보: ", answer['공공편익시설정보'],'\n\n 관련다른여행지를 추천받으시려면 \'다른여행지\'을 입력해주세요'))
            else:
                st.session_state.generated.append((user_input, "무슨말인지 모르겠습니다. 더 구체적인 정보를 입력해주세요.\n ex) '지역이름' '즐기고싶은 것' 등을 적어주세요."))

        st.session_state.past.append(user_input)


        
    for i in range(len(st.session_state['past'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        if len(st.session_state['generated']) > i:
            message(st.session_state['generated'][i], key=str(i) + '_bot')


def show_schedule_management():
    st.title("일정 관리")
    st.write("이곳은 일정 관리 화면입니다.")

    # Create or get the schedule_data list from the session state
    if 'schedule_data' not in st.session_state:
        st.session_state.schedule_data = []

    # Implement the schedule management functionality here
    # You can use Streamlit widgets like text inputs, buttons, and tables to interact with the user and display the schedule

    # Adding a schedule
    with st.form('add_schedule_form'):
        schedule_date = st.date_input('날짜')
        schedule_time = st.time_input('시간')
        schedule_description = st.text_input('일정 내용')
        add_schedule_button = st.form_submit_button('일정 추가')

    if add_schedule_button:
        # Create a dictionary with the schedule information
        new_schedule = {
            'Date': str(schedule_date),
            'Time': str(schedule_time),
            'Description': schedule_description
        }

        # Add the new schedule to the schedule_data list
        st.session_state.schedule_data.append(new_schedule)

        # Display a success message to the user
        st.success('일정이 추가되었습니다.')

    # Save the schedule data to a CSV file
    save_button = st.button('일정 저장')
    if save_button:
        df = pd.DataFrame(st.session_state.schedule_data)
        df.to_csv('schedule_data.csv', index=False)
        st.success('일정이 성공적으로 저장되었습니다.')

    # Load the schedule data from a CSV file
    load_button = st.button('일정 불러오기')
    if load_button:
        try:
            df = pd.read_csv('schedule_data.csv')
            st.session_state.schedule_data = df.to_dict('records')
            st.success('일정이 성공적으로 불러와졌습니다.')
        except FileNotFoundError:
            st.error('일정 데이터를 찾을 수 없습니다.')

    # Display the schedule data in a table
    if st.session_state.schedule_data:
        schedule_df = pd.DataFrame(st.session_state.schedule_data)
        
        # Add a dropdown for selecting schedules to delete
        selected_schedule_descriptions = st.multiselect('삭제할 일정 선택', schedule_df['Description'].tolist())
        
        # Delete the selected schedules from the schedule_data list
        st.session_state.schedule_data = [schedule for schedule in st.session_state.schedule_data if schedule['Description'] not in selected_schedule_descriptions]
        
        # Sort the schedule data by date
        schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
        schedule_df = schedule_df.sort_values('Date')
        schedule_df['Date'] = schedule_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Display the updated schedule data in a table
        st.table(schedule_df)

        # Delete selected schedules
        delete_schedule_button = st.button('일정 삭제')
        if delete_schedule_button:
            if selected_schedule_descriptions:
                st.session_state.schedule_data = [schedule for schedule in st.session_state.schedule_data if schedule['Description'] not in selected_schedule_descriptions]
                st.success('선택한 일정이 삭제되었습니다.')
            else:
                st.warning('삭제할 일정을 선택하세요.')
    else:
        st.write('일정이 없습니다.')



def main():
    with st.sidebar:
        choice = option_menu("Menu", ["홈", "여행 챗봇", "일정 관리"],
                             icons=['house', 'chat-dots', 'calendar-day'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "4!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
            "nav-link-selected": {"background-color": "#08c7b4"},
        }
        )

    if choice == "홈":
        show_home()
    elif choice == "여행 챗봇":
        show_travel_chatbot()
    elif choice == "일정 관리":
        show_schedule_management()

if __name__ == '__main__':
    main()
    
