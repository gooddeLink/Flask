# AI server
## 모자이크 (img_to_mosaic)
- 특징 기반의 Object 검출 알고리즘(Haar cascades)의 정면 얼굴 검출 파일(haarcascade_frontalface_default) 사용해 얼굴 검출 
- 검출한 얼굴 OpenCV 사용하여 모자이크 <br>
![image](https://user-images.githubusercontent.com/76814748/205234577-f7c342ed-0d36-4438-80dd-5c4887278b77.png)


## 수해인식 AI (이미지 분류)
- selenium 사용하여 <a href = "https://github.com/gooddeLink/Flask/blob/main/jupyter_notebooks/%EA%B5%AC%EA%B8%80_%EC%9D%B4%EB%AF%B8%EC%A7%80_%ED%81%AC%EB%A1%A4%EB%A7%81.ipynb"> 구글 이미지 크롤링</a>
- pretained된 <a href = "https://github.com/gooddeLink/Flask/blob/main/jupyter_notebooks/mobilenetv3_train.ipynb">MobileNetV3</a>, <a href = "https://github.com/gooddeLink/Flask/blob/main/jupyter_notebooks/resnet18_train.ipynb">ResNet18</a>, <a href = "https://github.com/gooddeLink/Flask/blob/main/jupyter_notebooks/pytorch_project_resnet50.ipynb">ResNet50</a> 모델 사용하여 transfer learning
- 성능이 가장 뛰어난 MobileNetV3 사용 


