# AI server
## 👨 모자이크 
- 특징 기반의 Object 검출 알고리즘(Haar cascades)의 정면 얼굴 검출 파일(haarcascade_frontalface_default) 사용해 얼굴 검출 
- 검출한 얼굴 OpenCV 사용하여 모자이크 <br>
![image](https://user-images.githubusercontent.com/76814748/205234577-f7c342ed-0d36-4438-80dd-5c4887278b77.png)


## 🌊 수해인식 AI
- selenium 사용하여 <a href = "https://github.com/gooddeLink/Flask/blob/main/jupyter_notebooks/%EA%B5%AC%EA%B8%80_%EC%9D%B4%EB%AF%B8%EC%A7%80_%ED%81%AC%EB%A1%A4%EB%A7%81.ipynb"> 구글 이미지 크롤링</a>
- 데이터 전처리 (<a href = "https://github.com/gooddeLink/Flask/blob/main/app.py">app.py</a>의 transform_image)
- pretained된 <a href = "https://github.com/gooddeLink/Flask/blob/main/jupyter_notebooks/mobilenetv3_train.ipynb">MobileNetV3</a>, <a href = "https://github.com/gooddeLink/Flask/blob/main/jupyter_notebooks/resnet18_train.ipynb">ResNet18</a>, <a href = "https://github.com/gooddeLink/Flask/blob/main/jupyter_notebooks/pytorch_project_resnet50.ipynb">ResNet50</a> 모델 사용하여 transfer learning
- 성능이 가장 뛰어난 MobileNetV3 사용 <br>


![image](https://user-images.githubusercontent.com/76814748/205241624-52127eda-52f6-4b42-a49b-417c1b6d59f3.png)
![image](https://user-images.githubusercontent.com/76814748/205244212-2d390bad-8e3e-482b-b111-1d9b2fc21129.png)
<br>
![image](https://user-images.githubusercontent.com/76814748/205241951-274dc51b-ddf4-41e1-8977-17323b0191f7.png)
![image](https://user-images.githubusercontent.com/76814748/205239648-88c38e5e-42cd-413d-a514-8d3bfbbb8df2.png)


