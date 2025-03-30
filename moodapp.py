import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
stemmer=PorterStemmer()

#Loading the pickle file
with open('LogisticRegression2.pickle','rb') as f:
    model=pickle.load(f)

#defining emotion

emotion_labels = {
    0: 'The person seems to be feeling sad. 😔',
    1: 'The person appears to be happy. 😊',
    2: 'The person seems to be in a joyful mood. 😄',
    3: 'The person appears to be angry. 😡',
    4: 'The person seems to be scared. 😨',
    5: 'The person appears to be very surprised. 😲'
}

#userinput pre-processing
def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^\w\s]","",text)
    text=" ".join(stemmer.stem(word) for word in text.split() if word not in stop_words)
    return text

st.title("Mood Detection App 🎭")
st.image('https://t4.ftcdn.net/jpg/03/08/12/65/360_F_308126573_fDAWMDCQVNzBsnXgqr2ldt8MrJBIun3P.jpg',width=450)
st.write("Please enter the text you received, and the model will predict the emotion behind it.")
st.write('*Please enter longer sentences to help the model make a more accurate prediction*')

user_input=st.text_input("Type your message here: ","")

if st.button("Predict Mood"):
    if user_input:
        cleaned_input=clean_text(user_input)
        predicted_label=model.predict([cleaned_input])[0]
        emotion=emotion_labels.get(predicted_label)

        st.subheader(f'Predicted Emotion: {emotion}')

    else:
        st.warning('Please enter a sentence to analyze.')
