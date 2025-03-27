import streamlit as st  
import pandas as pd
import json
import re
import easyocr
import cv2
import numpy as np
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize OCR Reader
reader = easyocr.Reader(["en"])

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "df_store" not in st.session_state or st.session_state.df_store is None:
    st.session_state.df_store = pd.DataFrame()
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

def OCR(image):
    """Extracts text from the uploaded image using EasyOCR."""
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    extracted_text = [text[1] for text in results] 
    return extracted_text

async def async_llm_response(extracted_text):
    """Sends extracted text to Gemini API for structured data extraction using asyncio."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyCcGqDgr0okwX3zivih9YRXciiCEixQx1c")
    
    prompt = f"""
    THE EXTRACTED TEXT IS FROM AN INVOICE.
    Extract the following structured data:
    - INVOICE (Invoice number)
    - DATE CREATED
    - VENDOR (Company name)
    - SALE TYPE
    - DELIVER TO
    - DESCRIPTION (as a list)
    - QUANTITY (Extract as an integer)
    - UNIT PRICE (Single item price)
    - EXTD PRICE or TOTAL (Total price)

    Return the data in **valid JSON format**.

    OCR Text:
    {extracted_text}
    """
    response = await asyncio.to_thread(llm.invoke, prompt)
    return response.content

def into_df(response):
    """Extracts JSON from response and appends it to a global DataFrame."""
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    json_string = json_match.group(1) if json_match else response

    try:
        extracted_json = json.loads(json_string)
    except json.JSONDecodeError:
        st.error("❌ Error: Invalid JSON format")
        return

    rows = []
    max_length = max([len(v) if isinstance(v, list) else 1 for v in extracted_json.values()])  

    for i in range(max_length):
        row = {key: (value[i] if isinstance(value, list) and i < len(value) else value)
               for key, value in extracted_json.items()}
        rows.append(row)

    new_df = pd.DataFrame(rows)
    new_df = new_df[new_df["UNIT PRICE"].notna() & (new_df["UNIT PRICE"] != "")]

    if new_df.empty:
        st.error("❌ No valid data to append.")
        return

    if "Quantity" in new_df.columns:
        new_df["Quantity"] = pd.to_numeric(new_df["Quantity"], errors="coerce").fillna(0).astype(int)

    st.session_state.df_store = pd.concat([st.session_state.df_store, new_df], ignore_index=True)

async def main():
    st.title("📄 Invoice Data Extraction")

    uploaded_file = st.file_uploader("Upload an invoice image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file

        action = st.radio("Do you want to process this image?", 
                      ["Yes, Process", "No, Upload Another"], 
                      index=None) 
        
        if not st.session_state.processed:
            if action == "Yes, Process":
                with st.spinner("Extracting text..."):
                    extracted_text = OCR(uploaded_file)
                
                if extracted_text:
                    st.success("✅ Text extracted successfully!")

                    with st.spinner("Processing structured data..."):
                        response = await async_llm_response(extracted_text)
                        into_df(response)

                    st.success("✅ Data processed and added to table!")
                    st.dataframe(st.session_state.df_store)
                    st.session_state.processed = True
                    
                    if not st.session_state.df_store.empty:
                        csv = st.session_state.df_store.to_csv(index=False).encode("utf-8")
                        st.download_button(label="📥 Download CSV", data=csv, file_name="extracted_data.csv", mime="text/csv")

        elif action == "No, Upload Another":
            st.session_state.uploaded_image = None
            st.session_state.processed = False
            st.session_state.df_store = pd.DataFrame()
            st.rerun()
        else:
            st.warning("🔄 Please Select an Option")

# Ensure the event loop runs properly
if __name__ == "__main__":
    asyncio.run(main())
