# Nutri-Vision AI: An End-to-End Food Nutrients Estimator Project 

This project is a deep learning-powered food recognition system that identifies popular Indian food items from an image (and customizes them using optional text) and provides an estimate of their nutritional content. The entire pipeline, from data collection and preprocessing to model training and deployment, is detailed below.

**Live Application:** https://nutri-vision-project-ouhmen7vgkadaszezkn6nx.streamlit.app/

---

## Dataset Creation Journey

The creation of this comprehensive food image dataset presented several technical challenges:

* **Initial Tool:** The project began with the `google-images-download` Python package (Joeclinton1 GitHub fork) for its perceived ease of use. This tool utilized Selenium and ChromeDriver for automated Google Image retrieval.
* **Initial Challenges:** The script frequently returned zero results, despite manual verification of image availability on Google Images for the same keywords.
* **Root Cause:** Updates to Google's website and image search behavior had rendered the initial scraping tool ineffective. Troubleshooting Selenium and ChromeDriver did not resolve the core issue.
* **Alternative Tool:** The `icrawler` Python package was adopted as a more robust alternative, offering better support for modern web page layouts and infinite scrolling, enabling more successful image crawling.
* **Key Learnings:** The dataset creation process underscored the importance of:
    * Verifying tool compatibility with target websites.
    * Utilizing actively maintained libraries.
    * Anticipating and managing data redundancy.
    * Considering long-term dataset maintenance.
* **Overall Outcome:** I had a dataset of 2000 high-quality images spanning 40 popular Indian food classes.

---

## Preprocessing

After the data collection phase, I moved on to preparing the dataset for training. This preprocessing step was critical to ensure better model performance and data consistency. Following is a detailed overview of my steps:

### 1. Manual Filtering and Cleanup

Before starting any automated preprocessing, I manually inspected the images and deleted any obviously poor-quality or irrelevant images. This helped reduce noise and prevented the model from learning from confusing or misleading samples.

### 2. Sequential Renaming of Images

To maintain a clean and consistent dataset structure, I wrote a simple script to rename all image files sequentially within each class folder (e.g., `1.jpg`, `2.jpg`, `3.jpg`…). This step was important because:

* It simplified dataset handling in later stages such as training and evaluation.
* It ensured that there were no filename conflicts or missing indices that could break scripts expecting orderly data.

### 3. Text Removal and Image Cleaning

Many of the collected images—especially those scraped from YouTube thumbnails, food blogs, and commercial packaging—contained overlaid text such as subtitles, product names, or branding. This introduced noise into the dataset, as models could easily overfit to non-food-related features like logos or captions rather than the visual characteristics of the food itself.

**Initial Experiment: OCR with Tesseract**

I initially explored OCR-based detection using Tesseract, hoping to locate text regions and then remove them via inpainting. However, this approach failed to generalize due to:

* Inconsistent text fonts, sizes, colours, and curvatures.
* Blending of text with complex or textured food backgrounds.
* Frequent missed detections or partial bounding boxes.

These limitations made OCR unsuitable for large-scale, automated cleaning.

**Final Solution: Deep Learning-Based Detection + Inpainting**

To address these issues, I built a complete preprocessing pipeline using the EAST text detector and OpenCV’s inpainting:

* **Text Localization:** Loaded the pretrained `frozen_east_text_detection.pb` model via OpenCV’s DNN module. Resized images to fit model constraints (multiples of 32), then passed them through the model to generate confidence scores and geometry maps for word-level detection.
* **Bounding Box Extraction:** Parsed geometry maps to compute rotated rectangles for detected text regions based on trigonometric calculations. Applied Non-Maximum Suppression (NMS) to filter overlapping or low-confidence boxes.
* **Masking + Inpainting:** Created a binary mask from the filtered bounding boxes, with slight padding to fully cover each text region. Used `cv2.inpaint` with the TELEA algorithm to naturally fill masked regions using surrounding pixel information, ensuring smooth blending and visual consistency.
* **Automation & Scaling:** Integrated this logic into a batch-processing loop that traverses all class folders and processes every image. Saved the cleaned, inpainted images back in place, replacing originals to simplify downstream use.

This robust approach allowed me to clean thousands of images automatically, removing distracting overlays while retaining realistic visual quality — a key factor for training a food recognition model that learns meaningful patterns from the actual dish content.

---

## Data Augmentation

Data augmentation is a critical step in enhancing the performance and generalization capability of machine learning models, particularly in image classification tasks. By artificially increasing the diversity of the training dataset, data augmentation helps the model become more robust to variations that may occur in real-world scenarios.

### Purpose of Data Augmentation

* **Increase Dataset Diversity:** Augmentation generates multiple altered versions of existing images, effectively expanding the dataset size without requiring additional data collection.
* **Prevent Overfitting:** By exposing the model to varied transformations of the input data, augmentation reduces the risk of the model memorizing specific training samples.
* **Enhance Generalization:** Models trained with augmented data are better equipped to handle real-world variations such as changes in orientation, lighting, and scale.

### Augmentation Techniques Employed

The following augmentation techniques were applied to the training dataset:

* **Padding and Resizing:** Images were padded to ensure a minimum size and then resized to a fixed resolution, helping standardize input dimensions while minimizing content loss.
* **Horizontal Flipping:** Food images were flipped horizontally with high probability to account for natural variations in plating and camera angles.
* **Slight Shifts, Zooms, and Rotations:** Controlled geometric transformations were applied to simulate different viewpoints and slight misalignments that can occur during image capture.
* **Subtle Brightness and Contrast Changes:** Adjustments were made to mimic different lighting conditions without distorting the natural appearance of the food.
* **Mild Color Adjustments:** Small variations in color tones, including hue, saturation, and value, were introduced to make the model more robust to camera and environment differences.
* **Minimal Blurring (only for selected categories):** Slight motion blur was applied sparingly to simulate real-world camera imperfections, but only for dishes where fine texture details are less critical.

These augmentations were carefully designed with category sensitivity in mind. For color-critical dishes like biryani, momos, and desserts, the transformations preserved the visual identity of the food. For less sensitive categories, broader variation was introduced to increase robustness. My dataset was multiplied to a total of 12k images using data augmentation, which is necessary to train a good deep learning model.

---

## Model Architecture and Training

My goal was to build a highly accurate model without having access to massive GPU resources. To achieve this, I used a combination of modern deep learning techniques.

### 1. Model Architecture: Fine-Tuning a Pre-trained Model

Instead of training a neural network from scratch, I used transfer learning. This involves taking a powerful, pre-trained model and adapting it to a new task.

* **Base Model:** I chose **EfficientNetB0**, a state-of-the-art model known for having a high accuracy while being lightweight. It was pre-trained on the massive ImageNet dataset, meaning it already knew how to recognize general visual features like edges, textures, and shapes.
* **Custom Layers:** I added my own custom layers on top of the `EfficientNetB0` base. This included a `GlobalAveragePooling2D` layer to process the features, followed by `Dropout` layers to prevent overfitting, and a final `Dense` layer with a softmax activation to classify images into one of my 40 food categories.

### 2. The Training Process

The final training workflow was carefully designed to find the best possible version of the model:

* **Hyperparameter Tuning:** I used an automated tool called Keras Tuner to efficiently search for the best model settings (like learning rate and dropout rate). The tuner found a highly effective combination very early in the process, which gave me a great starting point for final training.
* **Final Training:** Using the best settings found by the tuner, I trained the final model. I used key techniques like `EarlyStopping` (to stop training when the model is no longer improving) and `ReduceLROnPlateau` (to automatically lower the learning rate for fine-tuning), ensuring the model is both accurate and robust.

---

## The Debugging Journey: From Frustration to Success

Building the model was not a straightforward process. It involved significant debugging and taught me invaluable lessons about the realities of building machine learning pipelines.

### 1. The Corrupt File Mystery

My initial attempts at training were met with a frustrating, intermittent error. The model would train successfully for an epoch or two, then crash, complaining about an "unknown image file format."

* **The Challenge:** This was a huge roadblock that took three days to solve. I ran multiple verification scripts, but none could find any "corrupt" files. The real puzzle was why the training would run at all before failing.
* **The Breakthrough:** I realized the error only happened when the model tried to create a batch of images to process. This meant the issue wasn't a simple corrupt file but a subtle inconsistency in the data—some images were likely saved in Grayscale (1 channel) instead of RGB (3 channels). While valid images, they couldn't be stacked with the others in a batch.
* **The Solution:** I developed a targeted debugging script that used TensorFlow's own functions to check every single file for compatibility. This approach finally pinpointed the 15 files that were causing the crash. After removing them, the training process ran without interruption.

### 2. Uncovering Data Leakage

With the training finally running, I encountered a new puzzle: the model was achieving a validation accuracy of nearly 100% within just a few epochs, while the training accuracy lagged far behind.

* **The seemingly perfect result:** While exciting, a 100% validation accuracy is a major red flag for a common issue called data leakage.
* **The Diagnosis:** I reviewed my data pipeline and discovered the cause. My augmentation script was reading images from my validation set and saving the augmented versions into my training set. This meant the model was being trained on variations of the exact same images it was being tested on. It had effectively "seen the answers," leading to an artificially inflated score.
* **The Fix:** I corrected the pipeline by first creating a clean split of my original images into separate `train` and `val` folders. Only then did I apply data augmentation exclusively to the new, isolated `train` set. This ensured the validation set remained truly "unseen" data, providing a trustworthy measure of the model's performance.

### 3. Achieving a Robust Model

After resolving the data issues, I retrained the model using the proper workflow.

* **Final Performance:** The fine-tuned `EfficientNetB0` model achieved a final validation accuracy of **86.8%**. This is a strong and trustworthy result for a challenging 40-class problem, confirming the model's ability to generalize to new, unseen data.

---

## NLP-Powered Nutrient Estimation 

The project's key innovation lies in its multimodal approach, combining computer vision for food recognition with Natural Language Processing (NLP) for a richer, more accurate nutritional analysis. While the deep learning model identifies the food item from an image, the NLP pipeline processes optional text inputs to refine the prediction and calculate specific nutrient values. These modifiers will then be applied to the output retrieved from a custom nutrient database for a more personalised and relevant nutrient estimation.

### The NLP Pipeline:

1.  **Text Input Processing:** The user's text input—which might include the food name, portion size (e.g., "half a cup"), or preparation details (e.g., "biryani with extra rice")—is tokenized and preprocessed. This involves converting text into a structured, machine-readable format.
2.  **Entity Recognition and Extraction:** The NLP model, typically a custom-trained named-entity recognition (NER) model, identifies key entities. It extracts:
    * **Food Items:** Matches the user's text to a known food item or category.
    * **Quantity/Unit:** Recognizes portion sizes like small, standard/medium or large serving.
    * **Descriptors:** Captures words that modify the food item, such as "fried," "grilled," or "extra."
3.  **Nutrient Database Lookup:** The extracted entities are used to query a comprehensive nutritional database. The system retrieves standard nutritional information for the identified food, adjusted for the specified quantity. For example, if the model identifies "one cup of rice," it queries the database for the nutritional value of that specific portion.
4.  **Customization and Calculation:** This is where the true power of the NLP-Vision integration is realized. The system uses the text descriptors to customize the nutritional estimate. For instance, if the image-based prediction is "pizza," and the user adds the text "homemade wheat with extra cheese" the system uses NLP to calculate the effect of all these modifiers on the nutritional content of the food. If a user inputs "mutter paneer, large size", the system can provide an estimate that is proportional to a large serving size.
5.  **Final Output Generation:** The NLP-refined nutritional data is then presented to the user, providing a more personalized and accurate estimate than a simple image-based recognition system could provide alone.

This integration of NLP transforms the project from a simple food classifier into an intelligent nutrient estimator. It empowers the user to provide context and customization, leading to a much more accurate and helpful application.

---

## Creating a Streamlit UI

To make the trained food classification model easily accessible and interactive, I developed a simple user interface using Streamlit. This allows anyone to upload a food image and instantly see the model's prediction.

The core functionality of the Streamlit UI includes:

* **Image Upload:** Users can upload an image of food in common formats (e.g., JPG, PNG).
* **Prediction Display:** Once an image is uploaded, the model processes it, and the top predicted food category is displayed to the user.
* **Text input (Optional):** Along with (or even without) the image, the users can type text inputs of the food, or the portion size or customizations for accurate predictions relevant to their food.

This Streamlit application makes it easy to showcase the capabilities of the trained food classification model and allows for quick and intuitive interaction.

---

## Conclusion

This project was a deeply enriching experience that combined real-world challenges in data collection, image preprocessing, model training, debugging, and user-facing deployment. From overcoming broken scrapers and image inconsistencies to solving subtle bugs like data leakage and grayscale incompatibilities, each hurdle taught me to think systematically and engineer robust solutions. The result is a high-performing, scalable food classification system powered by deep learning and accessible via an intuitive Streamlit interface. More than just a technical achievement, this journey has strengthened my end-to-end machine learning skills and prepared me for tackling even more ambitious applied AI projects in the future.
