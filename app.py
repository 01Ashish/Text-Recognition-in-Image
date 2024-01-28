import streamlit as st
import cv2
import numpy as np
import easyocr
import pyttsx3
import matplotlib.pyplot as plt
import pytesseract

# Load the OCR model
reader = easyocr.Reader(['en'])
pytesseract.pytesseract.tesseract_cmd='D:\\ocr\\tesseract.exe'
# Function to recognize text
def recognize_text(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_easyocr = reader.readtext(img)
    result_pytesseract = pytesseract.image_to_string(img,lang='eng')
    return result_easyocr, result_pytesseract

# Function to overlay text on image
def overlay_ocr_text(img_path, save_name):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dpi = 80
    fig_width, fig_height = int(img.shape[0] / dpi), int(img.shape[1] / dpi)
    fig, axarr = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    axarr[0].imshow(img)

    result_easyocr, result_pytesseract = recognize_text(img_path)

    for (bbox, text, prob) in result_easyocr:
        if prob >= 0.5:
            print(f'Detected text: {text} (Probability: {prob:.2f})')
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)
            cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=8)

    axarr[1].imshow(img)
    axarr[1].set_title('Pytesseract OCR')
    axarr[1].imshow('Off')
    
    result_pytesseract = result_pytesseract.replace('\n','')
    axarr[2].text(0, 0.5, result_pytesseract, fontsize=16)
    axarr[2].set_title('Pytesseract Text')
    axarr[2].axis('off')

    plt.savefig(f'./output/{save_name}_overlay.jpg', bbox_inches='tight')

# Function to convert text to speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)
    engine.say(text)
    engine.runAndWait()

# Streamlit app
def main():
    st.title("Text Recognition and Speech Synthesis")

    # File uploader to allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption='Uploaded Image')
        
        # Save the uploaded image to a temporary file
        temp_img_path = "./temp_image.jpg"
        cv2.imwrite(temp_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Recognize text
        result_easyocr, result_pytesseract = recognize_text(temp_img_path)

        # Display the recognized text
        sentence = ''
        for (bbox, text, prob) in result_easyocr:
            sentence += f'{text} '
        st.subheader('Recognized Text From EasyOCR')
        st.write(sentence)
        
        # Speak the text
        st.subheader('Speak')
        st.button("Speak Text",key='speak easy ocr', on_click=lambda: speak_text(sentence))
        
        st.subheader('Recognized Text From PyTesseract')
        st.write(result_pytesseract)

        # Speak the text from PyTesseract
        st.subheader('Speak')
        st.button("Speak Text",key = 'speak pytesseract',on_click=lambda: speak_text(result_pytesseract))

if __name__ == '__main__':
    main()
