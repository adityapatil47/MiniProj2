**1. clone the repo:**   
  - `git clone https://github.com/adityapatil47/MiniProj2.git`    
  - `cd MiniProj2`
  - `mkdir -p resources/data/emotions resources/data/faces`
- `mkdir -p resources/models/emotions resources/models/faces`

**2. create and activate env:**

- > **NOTE**: use python 3.8 for environment

  `python -m venv venv`  
  `source venv/Scripts/activate`

- confirm env is activated  
   `which python`
- download all requirements:  
   `pip install -r requirements.txt`  
  [cuDNN v7.5.1 (April 22, 2019), for CUDA 10.1](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse751-101)  
  [CUDA Toolkit 10.1 (Update 2)](https://developer.nvidia.com/cuda-10.1-download-archive-update2)

**3. create test and train data:**

- _[download fer2015 dataset](https://www.kaggle.com/datasets/deadskull7/fer2013)_
  - save dataset to path `\resources\data`
  - Now run the python notebook [data_analysis.ipynb](src/emotions/dataset_analysis.ipynb)

**4. Train emotion recognition:**

> **NOTE**: The working directory is `src`.

- Run:  
  `python train_emotions.py --epochs 30 --batch_size 32 --lr 0.001 --decay 1e-6 --dropout 0.3 --l2 0.01 --kernel 3`

**6. Dataset and Face acquisition**

- donwload [haarcascade_frontalface_default OpenCV dataset](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml)

  - save to path resources/models/

- collect realtime data:  
   `python face_acquisition.py --camera 0 --name "<YOUR_NAME>" --num_photos 100`

- train faces  
  `python train_faces.py`

**7. Optimize emotion recognition:**

- Run:  
  `python optimize_emotions.py --n_trials 50`

**8. Evaluate:**

- Run:  
  `python evaluate_emotions.py`

**9. display:**

- For Webcam  
  `python display.py --camera 0`
