Object Tracking with Deep Learning
This project demonstrates how to create a deep learning model to track an object in your hand and determine its movement direction (left, right, up, down).

Setup
Prerequisites
Conda
Python 3.8
Installation
1. Clone the repository:
2. Create and activate Conda environment:


conda create --name OBJTRACK python=3.8
conda activate OBJTRACK

3. Install dependencies:


pip install -r requirements.txt

Usage
Preprocess the Data
1. Collect video data of the object moving in different directions and save the frames as images in the data/raw directory.

2. Run the preprocessing script:


python src/preprocess.py

Train the Model
Train the model using the preprocessed data:


python src/model.py

Track the Object
1. Run the object tracking script to track the object in real-time:

python src/track.py

Future Improvements
Cloud deployment
Edge deployment
Model accuracy improvements


License
This project is licensed under the MIT License.