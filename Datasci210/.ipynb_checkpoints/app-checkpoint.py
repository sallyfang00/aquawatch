import streamlit as st
from PIL import Image
import torch
import tensorflow as tf
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from streamlit.components.v1 import html

# # Load Mask-RCNN model
# def get_maskrcnn_model(num_classes):
#     model = maskrcnn_resnet50_fpn(pretrained=True)
#     # ... (replace or modify the classifier for the number of classes)
#     model.eval()
#     return model

# # Preprocess image for Mask-RCNN
# def preprocess_maskrcnn_image(img):
#     transform = T.Compose([T.ToTensor()])
#     img = transform(img).unsqueeze(0).to(device)  # Assuming 'device' is defined
#     return img

# # Preprocess image for CNN
# def preprocess_cnn_image(img):
#     img = img.resize((224, 224))
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     #img /= 255.0  # I don't think we normalize pixel values to between 0 and 1?
#     return img

# # Load CNN model
# def get_cnn_model():
#     # ... (your model loading code)
#     return model


# # Page 1: Main Page
# def main_page():
#     st.title("Welcome to Aquawatch 💧")
#     st.subheader("The Algae Bloom Toxicity Detection App")
#     st.text("This web app detects the toxicity level of algae in water based on images.")
#     st.text("Please see below for examples of the 3 tiers of toxins.")


#     # Display images of the 3 tiers
#     col1, col2, col3 = st.columns(3)
#     col1.image("data/no_advisory.jpg", caption='No Advisory', use_column_width=True)
#     col2.image("data/warning.jpg", caption='Caution', use_column_width=True)
#     col3.image("data/danger.jpg", caption='Danger', use_column_width=True)

#     st.text("We classified the tiers based on the following categorization:")

#     data = {
#         "Category": ["No advisory", "Caution", "Danger"],
#         "Description": [
#             "Total Microcystins: <0.8μg/L",
#             "Total Microcystins: 0.8μg/L to <20μg/L",
#             "Total Microcystins: ≥20μg/L"
#         ]
#     }

#     # Create a DataFrame
#     df = pd.DataFrame(data)

#     # Create a table
#     st.table(df)


# # Page 2: Detection Page
# def detection_page():
#     st.title("Algae Bloom Toxicity Detection")
#     st.text("Upload an image to detect the toxicity level of algae.")

#     uploaded_file = st.file_uploader("Please upload your image here")
#     st.text("Please ensure that you upload an image capturing only the water body.")
#     st.text("Avoiding reflections on the water will lead to better identification of algae toxicity levels.")

#     if uploaded_file is not None:
#         # Show the uploaded image
#         img = Image.open(uploaded_file)
#         st.image(img, caption='Uploaded Image.', use_column_width=True)

#         # # Load Mask-RCNN model
#         # maskrcnn_model = get_maskrcnn_model(num_classes=1)
#         #
#         # # Preprocess image for Mask-RCNN
#         # img_maskrcnn = Image.open(uploaded_file).convert("RGB")
#         # preprocessed_img_maskrcnn = preprocess_maskrcnn_image(img_maskrcnn)
#         #
#         # # Apply Mask-RCNN to get the mask
#         # with torch.no_grad():
#         #     prediction = maskrcnn_model(preprocessed_img_maskrcnn)
#         #
#         # # Assuming you get masks from the prediction, use it to preprocess the image for CNN
#         # # For example, you can apply the mask to the original image
#         # mask = prediction[0]['masks'][0, 0].cpu().numpy()
#         # img_cnn = np.multiply(img_maskrcnn, np.stack([mask, mask, mask], axis=-1))
#         #
#         # # Preprocess image for CNN
#         # img_cnn = preprocess_cnn_image(img_cnn)
#         #
#         # # Load CNN model
#         # cnn_model = get_cnn_model()
#         #
#         # # Make predictions using the CNN model
#         # cnn_prediction = cnn_model.predict(img_cnn)
#         #
#         # # Use the predictions for further display or analysis
#         # st.write(f"Predicted CNN Output: {cnn_prediction}")

# Sidebar with links to switch between pages
# selected_page = st.sidebar.radio("Select Page", ["Main Page", "Detection Page"])

# # Display the selected page
# # if selected_page == "Main Page":
# #     main_page()
# # elif selected_page == "Detection Page":
# #     detection_page()

# def nav_page(page_name, timeout_secs=3):
#     nav_script = """
#         <script type="text/javascript">
#             function attempt_nav_page(page_name, start_time, timeout_secs) {
#                 var links = window.parent.document.getElementsByTagName("a");
#                 for (var i = 0; i < links.length; i++) {
#                     if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
#                         links[i].click();
#                         return;
#                     }
#                 }
#                 var elasped = new Date() - start_time;
#                 if (elasped < timeout_secs * 1000) {
#                     setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
#                 } else {
#                     alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
#                 }
#             }
#             window.addEventListener("load", function() {
#                 attempt_nav_page("%s", new Date(), %d);
#             });
#         </script>
#     """ % (page_name, timeout_secs)
#     html(nav_script)
    
# if st.button("< Prev"):
#     nav_page("main_page")
# # if st.button("Next >"):
# #     nav_page("Bar")


def nav_page(page_name, timeout_secs=3):
    nav_script = f"""
        <script type="text/javascript">
            function navigateToPage(page_name) {{
                window.location.href = window.location.origin + "?page=" + page_name;
            }}
        </script>
    """
    st.markdown(nav_script, unsafe_allow_html=True)
    st.button("Continue", on_click="navigateToPage('another_page')")

def main():
        
    st.title("Welcome to Aquawatch 💧")
    st.subheader("The Algae Bloom Toxicity Detection App")
    st.text("This web app detects the toxicity level of algae in water based on images.")
    st.text("Please see below for examples of the 4 tiers of toxins.")


    # Display images of the 3 tiers
    col1, col2, col3 = st.columns(3)
    col1.image("data/no_advisory.jpg", caption='No Advisory', use_column_width=True)
    col2.image("data/warning.jpg", caption='Caution', use_column_width=True)
    col3.image("data/danger.jpg", caption='Danger', use_column_width=True)

    st.text("We classified the tiers based on the following categorization:")

    data = {
        "Category": ["No advisory", "Caution", "Danger"],
        "Description": [
            "Total Microcystins: <0.8μg/L",
            "Total Microcystins: 0.8μg/L to <20μg/L",
            "Total Microcystins: ≥20μg/L"
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create a table
    st.table(df)
    
    page_name = st.experimental_get_query_params().get("page", [""])[0]
    
    if page_name == "another_page":
        st.write("This is Page 1 content.")
    nav_page(page_name)

if __name__ == "__main__":
    main()