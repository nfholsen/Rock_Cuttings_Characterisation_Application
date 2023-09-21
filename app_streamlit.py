import pandas as pd
from PIL import Image
import random
import streamlit as st
from sklearn.metrics import accuracy_score

from helpers import plot_confusion_matrix
import matplotlib.pyplot as plt


N = 4 # Number of samples per class min = 1 max = 20
n_samples = N * 5
st.set_page_config(layout="wide")


def run():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.markdown("<style>.row-widget.stButton {text-align: center;}</style>", unsafe_allow_html=True)

    image_style = """<style>.center-image {display: flex; justify-content: center;}</style>"""
    st.markdown(image_style, unsafe_allow_html=True)
    
    if 'counter' not in st.session_state:
        st.session_state.counter = 0

    if 'images' not in st.session_state:
        st.session_state.images = []

    if 'labels' not in st.session_state:
        st.session_state.labels = []

    if 'preds' not in st.session_state:
        st.session_state.preds = []

    if 'random_idx' not in st.session_state:
        st.session_state.random_idx = random.sample(range(0,n_samples), n_samples)

    def next_image(id_pressed):

        label = labels[st.session_state["random_idx"][st.session_state["counter"]]]

        st.session_state.preds.append((id_pressed,label))
        st.session_state["counter"] += 1        

    def restart():
        st.session_state["counter"] = 0
        st.session_state.preds = []

    ### Sidebar

    # Create dictionary to map test names to CSV file paths - First : example, Second : pred
    test_csv_paths = {
        'Lab/Lab': ['config/lab/train_test_train_mar.csv','config/lab/train_test_test_mar.csv'],
        'Lab/Borehole': ['config/lab/train_test_train_mar.csv','config/borehole/train_test_test_mar.csv'],
        'Borehole/Borehole': ['config/borehole/train_test_train_mar.csv','config/borehole/train_test_test_mar.csv'],
        'Borehole/Lab': ['config/borehole/train_test_train_mar.csv','config/lab/train_test_test_mar.csv']
    }

    ### Explanation ###
    st.sidebar.title('Rock Cuttings Classification App')
    st.sidebar.write('This app is used to produce results for the paper entilted Micro CT characterization of rock cuttings with Deep Learning.')
    st.sidebar.write('It allows for the user to classify rock cuttings obtained when drilling boreholes. There are 5 types of rock, 20 for each class for a total of 100 samples to classify for each test. The rocks are the following')
    st.sidebar.table(
        {
            'BL':'Bioclastic limestone',
            'GN':'Gneiss',
            'ML':'Micritic limestone',
            'MS':'Molassic sandstone',
            'OL':'Oolithic limestone'
        }
    )

    ### Test selection ###
    test_name  = st.sidebar.selectbox('Select test', options=list(test_csv_paths.keys()),on_change=restart)

    ### Settings##
    col1, col2 = st.sidebar.columns(2)
    with col1:
        download_but = st.download_button('Download results',file_name=f'{test_name}.csv',data=pd.DataFrame(st.session_state.preds,columns=['Prediction','Label']).to_csv().encode('utf-8'))
    with col2:
        restart_but = st.button('Restart', on_click=restart)

    with st.sidebar.expander("How to proceed"):
        st.write(
            '1. Select a test (it will start automatically)\n2. Complete the test\n3. when the last image is reached you can download the results and send the file to nils.olsench@gmail.com')

    st.sidebar.write('More on : https://github.com/nfholsen/Rock_Cuttings_Characterisation')

    if st.session_state["counter"] < n_samples:

        # Data
        # Images
        df_images = pd.read_csv(test_csv_paths[test_name][1], index_col=0).groupby(by=['Label']).sample(n=N,random_state=0).reset_index(drop=True)
        images = df_images['Paths_Test'].tolist()
        labels = df_images['Label'].tolist()
        # Examples
        df_examples = pd.read_csv(test_csv_paths[test_name][0], index_col=0)
        examples = df_examples['Paths_Test'].tolist()

        ### Main page
        st.write(f'You have chosen the test : {test_name}')

        col1_main, col2_main = st.columns(2, gap='medium')
        
        ### Column 1 - Predictions
        col1_main.write('Cuttings to predict')

        img_path = images[st.session_state["random_idx"][st.session_state["counter"]]]
        image = Image.open(img_path)
        image = image.resize((600, 600))
        col1_main.image(image,caption=f"Image {st.session_state['counter']+1}/{n_samples}")

        # Buttons for predictions
        col1, col2, col3, col4, col5 = col1_main.columns([1,1,1,1,1])

        with col1:
            st.button(f'Bioclastic limestone',key=f'but1',on_click=next_image,kwargs={'id_pressed':0})

        with col2:
            st.button(f'Gneiss',key=f'but2',on_click=next_image,kwargs={'id_pressed':1})

        with col3:
            st.button(f'Micritic limestone',on_click=next_image,kwargs={'id_pressed':2}, )

        with col4:
            st.button(f'Molassic sandstone',key=f'but4',on_click=next_image,kwargs={'id_pressed':3})

        with col5:
            st.button(f'Oolithic limestone',key=f'but5',on_click=next_image,kwargs={'id_pressed':4})

        ### Column 2 - Examples
        col2_main.write('Examples for Lab cuttings')
        col1, col2, col3, col4, col5 = col2_main.columns(5)

        for i, (col, label) in enumerate(zip([col1, col2, col3, col4, col5],['BL','GN', 'ML', 'MS', 'OL'])):
            with col:
                st.write(label)
                for j in range(5):
                    col.image(Image.open(examples[i*10 + j]).resize((400, 400)))

    else:
        st.warning('You reached the last image, click on send, restart or select an other test')

        df_results = pd.DataFrame(st.session_state.preds,columns=['Prediction','Label'])


        st.write(f"Here are your results for the test : {accuracy_score(df_results['Label'], df_results['Prediction'])* 100:2} %")

        df_results = pd.DataFrame(st.session_state.preds,columns=['Prediction','Label'])

        fig = plot_confusion_matrix(df_results['Label'], df_results['Prediction'], ['BL','GN', 'ML', 'MS', 'OL'])

        st.pyplot(fig)

if __name__ == '__main__':
    run()