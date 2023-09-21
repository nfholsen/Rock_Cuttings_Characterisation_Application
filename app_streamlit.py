import pandas as pd
from PIL import Image
import random
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


N = 20

st.set_page_config(layout="wide")


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred,normalize="true") if normalize else confusion_matrix(y_true, y_pred)

    cm_all = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
    bottom, top = ax.get_ylim()
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim(bottom, top)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        ax.text(cm.shape[0] + 0.5, i, cm_all.sum(axis=1)[i],
                ha="center", va="center")
    ax.text(cm.shape[0] + 0.5, -1, "N Samples",
            ha="center", va="center")
    fig.tight_layout()
    plt.grid(False)
    return fig



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
        st.session_state.random_idx = random.sample(range(0,N), N)

    def next_image(id_pressed):

        label = labels[st.session_state["random_idx"][st.session_state["counter"]]]

        st.session_state.preds.append((id_pressed,label))

        print(st.session_state.preds)

        st.session_state["counter"] += 1

    def restart():
        st.session_state["counter"] = 0
        st.session_state.preds = []

    ### Sidebar

    # Create dictionary to map test names to CSV file paths - First : example, Second : pred
    test_csv_paths = {
        'Lab/Lab': ['data/lab/train_test_train_mar.csv','data/lab/train_test_test_mar.csv'],
        'Lab/Borehole': ['data/lab/train_test_train_mar.csv','data/borehole/train_test_test_mar.csv'],
        'Borehole/Borehole': ['data/borehole/train_test_train_mar.csv','data/borehole/train_test_test_mar.csv'],
        'Borehole/Lab': ['data/borehole/train_test_train_mar.csv','data/lab/train_test_test_mar.csv']
    }

    ### Explanation ###
    st.sidebar.write('This app allows for the user to classify rock cutings. There are 5 types of rock :')
    st.sidebar.table(
        {
            'BL':'Bioclastic limestone',
            'GN':'Gneiss',
            'ML':'Micritic limestone',
            'MS':'Molassic sandstone',
            'OL':'Oolithic limestone'
        }
    )
    st.sidebar.write('How to proceed : select a test (it will start automatically), complete the test, when the last image is reached you can download the results and send it to nils.olsench@gmail.com')

    ### Test selection ###
    test_name  = st.sidebar.selectbox('Select test', options=list(test_csv_paths.keys()),on_change=restart)

    ### Settings##
    col1, col2 = st.sidebar.columns(2)
    with col1:
        download_but = st.download_button('Download results',file_name=f'{test_name}.csv',data=pd.DataFrame(st.session_state.preds,columns=['Prediction','Label']).to_csv().encode('utf-8'))
    with col2:
        restart_but = st.button('Restart', on_click=restart)

    if st.session_state["counter"] < N :

        # Data
        # Images
        df_images = pd.read_csv(test_csv_paths[test_name][1], index_col=0)
        images = df_images['Paths_Test'].tolist()[:N]
        labels = df_images['Label'].tolist()[:N]
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
        col1_main.image(image,caption=f"Image {st.session_state['counter']+1}/{N}")

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