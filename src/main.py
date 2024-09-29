import streamlit as st
from reader.qwen_reader import QwenOCRReader
from pathlib import Path
from PIL import Image
import numpy as np

# max_width = 800
# max_height = 800
qwen_reader = QwenOCRReader()

def resize_image(image, max_height=800, max_width=800):
   """Resize the image only if it exceeds the specified dimensions."""
   original_width, original_height = image.size
   
   # Check if resizing is needed
   if original_width > max_width or original_height > max_height:
      # Calculate the new size maintaining the aspect ratio
      aspect_ratio = original_width / original_height
      if original_width > original_height:
         new_width = max_width
         new_height = int(max_width / aspect_ratio)
      else:
         new_height = max_height
         new_width = int(max_height * aspect_ratio)
      
      # Resize the image using LANCZOS for high-quality downscaling
      return image.resize((new_width, new_height), Image.LANCZOS)
   else:
      return image

logo_path = Path(__file__).parent / "images/kumon-method-logo.svg"
st.logo(image=str(logo_path.absolute()))
st.sidebar.markdown("Kumon AI Assistant")
st.title(":blue[_*Kumon AI*_]")
USER = "user"
ASSISTANT = "assistant"

img_file_buffer = st.file_uploader('Upload a PNG/JPEGE image')
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    # display_image = image.resize((600, 800), Image.LANCZOS)
    display_image = resize_image(image)
    # img_array = np.array(image)
    st.image(display_image)
    # st.image(image, use_column_width=True)
    # result = qwen_reader.reader(image)
    # st.text(qwen_reader.reader(image))
    # st.markdown(result)
    if prompt := st.chat_input("Ask a question from the image..."):
      st.chat_message(USER).write(prompt)
      # results = qwen_reader.reader(image, prompt)
      results = qwen_reader.reader(display_image, prompt)
      with st.chat_message(ASSISTANT):
         for result in results:
            output = result.replace('\\n', '\n')
            st.write(output)