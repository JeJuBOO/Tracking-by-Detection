# Tracking-by-Detection
\[논문 리뷰\]  
[**High-Speed Tracking-by-Detection Without Using Image Information**](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf)  
Erik Bochinski, Volker Eiselein and Thomas Sikora  
Communication System Group, Technische Universitat Berlin  
Einsteinufer 17, 10587 Berlin

## Abstract
이 논문은 영상의 프레임 증가에 따라 기존 알고리즘과 견줄 수 있는 훨신 간단한 알고리즘을 제시하고 있다.

## 1. Introduction
### **객체 추적의 사용 사례** 
  객체 추적은 트래픽 분석, 스포츠 또는 포렌식과 같은 분석 시스템에 정보를 제공하거나, 차량의 번호판 인식, 얼굴 인식과 같은 곳에서 사용되고 있다.
### **한계점 제시** 
  특히,  tracking-by-detection systems의 일반적인 어려움은 실시간으로 진행될 경우 false positive 와 missed detections이 나타날 수도 있다는 것이다. 또한 여러 객체들이 서로 교차하며 경계가 모호해질 경우 더 많은 문제들이 발생한다.
### **한계점을 해결하는 다양한 방법들이 있고 최근 연구는 상당히 발전되어 있 그 중에서...**
 [참고된 논문\[8\]](#5-references)에서 소개된 아이디어를 기반하여 단순한 tracking 접근에 대해서 작성하고 이는 최근 연구된 detector들 보다 더 단순한 tracking 접근으로 성공적인 결과를 만들었다.  
 하지만 제안하는 tracking 알고리즘이 모든 경우에 있어서 필요한 것은 아니라는 것을 확인 해야한다.  
### **논문의 활용성** 
  제안하는 방법은 다른 방법들을 아우르는 단순한 방법으로 사용될 수 있고, 추가되는 tracking 알고리즘에 있어 기준이 될수 있다.
![image](https://user-images.githubusercontent.com/71332005/224903580-f4a7b85d-3391-4bc6-8a05-57a4e9c51d8d.png)

## 2. Method
### **가정**   
- 매 프레임마다 검출기(detector)는 추적(track)할 모든 객체를 검출(detect)한다. 즉, detection에 "gap"이 아예 없거나 거이 없다.
- 충분히 높은 프레임을 사용할 때 일반적으로 발생하는 overlap IOU가 충분히 높다고 가정한다.
  + overlap IOU : 간단하게 현재 프레임의 객체와 다음 프레임 객체가 겹쳐있는 정도.   

$$IOU(a,b)=\frac{Area(a)\bigcap Area(b)}{Area(a)\bigcup Area(b)}$$  

- 위 두가지 가정이 모두 충족된다면, 이미지 정보 없이도 수행될 수 있다.  
- **이 논문은 특정한 임계값* $\sigma_{IOU}$ *를 만족할 경우 이전 프레임의 detection결과와 가장 높은 IOU값을 같는 detection결과를 연관 시킴으로서 간단한 IOU Tracker를 제시한다.**

### **성능향상**
- $t_{min}$보다 길이가 짧거나 $\sigma_h$이상의 점수를 가진 detection이 하나도 없는 track을 제거함으로 성능을 향상시킬 수 있다.(짧은 track들은 일반적으로 false positive를 야기하고 출력에 혼란(clutter)을 추가하기 때문에)
- track에 적어도 하나 이상의 높은 점수를 가진 detection을 갖도록 요구하면, track의 완성도를 위한 낮은 점수의 detection에해대해 이점을 가진다.

- 논문에서 제안한 **알고리즘 1**이다. 
![Algirithm_1](https://user-images.githubusercontent.com/71332005/224783065-8a86d68e-d903-4abd-a078-f17df4b41ee3.png)
  + $D_f$: frame f에서의 detection
  + $d_j$: 해당 frame에서의 j번째 detection
  + $T_a$: 활성화된(actived) track
  + $T_f$: 완료된(finished) track
  + $F$: frame 수

- 알고리즘의 5번 라인을 보면 가장 잘 일치하지만 할당되지 않은 detection만 track에 추가할 후보가 된다.
  + 이는 반드시 detection($D_f$)와 Track($T_a$)사이의 연관성으로 이어지진 않지만, 해당 프레임에서 모든 IOU들의 합을 최대화 하는 Hungatian algorithm을 적용함 으로 해결 할 수 있다.
  + 일반적인 $\sigma_{IOU}$는 검출기의 비-최대 억제(non-maxima suppression)를 위한 IOU 임계값과 같은 범위에 있으므로 가장 높은 IOU를 선택하는 것은 합리적인 huristic한 방법이다.

### 제안한 알고리즘의 이점
  + 제안한 방법은 다른 최신의 tracker들과 비교해도 매우 낮은 복잡도를 가지고 있다.
  + 프레임의 시각적 정보는 사용되지 않아도 되기 때문 감지 수준에 대한 간단한 필터링 방법으로도 볼 수 있다.
  + 실시간으로 tracker가 최신 검출기(detector)와 함께 사용되는 경우, 최신 검출기에 비해 tracker의 계산 비용은 무시할 수 있는 수준이다.
  + tracker를 단독으로 수행해도 100Kfps를 초과하는 프레임율을 쉽게 달성할 수 있다.
  + tracker의 속도적 이점으로 출력을 이미지나 움직임 정보를 사용하여 연결할 수 있는 tracklets으로 고려함으로 detector의 결과에 tracking 구성 요소를 추가할 수 있다. 

## 3. Experiments  
- **DETRAC dataset를 사용하여 제안된 추적기의 성능을 조사했다.** DETRAC dataset는 차량 검출 및 추적을 목적으로 10시간 이상의 영상으로 구성된 데이터셋이다.
- **detector는 CompACT[[5]](#5-references), R-CNN[[10]](#5-references), ACF[[7]](#5-references), VGG16 1-3-5 모델을 이용한 Evolving Boxes detector(EB)[[16]](#5-references)등을 사용한다.**
- DPM[[9]](#5-references) 는 너무 부정확하기 때문에 제안한 tracker 적합하지 않아 결과에 포함하지 않았다.

### [**분류성능평가지표**](#분류성능평가지표classification-evaluation-metrics)  
- UA-DETRAC evaluation protocol을 사용하여 평가를 수행했다.
  + 이는 P-R curve을 계산하기 위해, 서로 다른 detection 점수의 임계값인 $\sigma_l$을 tracking에 여러번 적용하는 것이다.
  + 이 곡선을 통해 일반적인 CLEAR MOT matric[[14]](#5-references)이 계산되며, 최종적인 점수는 AUC로 구성되고 모든 detector의 임계값 $\sigma_l$에 대한 성능을 고려한다.(자세한 내용 참고[[17]](#5-references))
  + 이는 임계값 $\sigma_h$ 설정에 영향을 미치지 않고 낮은 점수의 detection를 사용하는 것에 영향을 미친다. 
  + 일반적으로 [8]에 따라 점수가 낮은 detection의 수가 많을수록 본 논문이 제안하는 tracking 성능이 향상된다고 볼 수 있다.

- 구현을 오직 파이썬으로 진행했고, 어떠한 성능 최적화도 수행하지 않았다.  

### **Parameter Search**

![Table 1](https://user-images.githubusercontent.com/71332005/224924926-3ed701de-40b4-4a2f-8ba3-baf835752c92.png)

![Table 2](https://user-images.githubusercontent.com/71332005/224920606-a9805588-4113-4020-aa00-d5f0c50b6ec3.png)

![Figure 2](https://user-images.githubusercontent.com/71332005/224920837-ea2636b1-105c-4ec0-b8d2-3f9288749b2d.png)

- $\sigma_{IOU}$, $\sigma_h$, $t_{min}$에 대한 최적의 parameters는 각 detector에 대한 훈련 데이터셋을 가지고 grid search를 수행하여 결정했다.
- 모든 detection 점수들은 $\sigma\in \[0.0;1.0\]$로 정규화 되었으나, 여전히 detector마다 다르게 분포되어있다. 따라서, $\sigma_h$의 범위도 다르게 선택되어야 한다.

- Table 2의 3가지 parameters의 범위 내의 모든 조합을 평가하기 위해 각 detector 당 64번 실행시킨 결과를 Figure 2로 나타냈다. UA-DETRAC 챌린지의 기본 측정기법(metric)인 PR-MOTA metric으로 최상의 구성을 선택한다.

- 각 detector에 대한 최상의 결과와 구성을 Table 3에서 볼 수있다.
![Table 3](https://user-images.githubusercontent.com/71332005/224925563-1adb1ebf-f21b-4edd-b5d2-188f2ce1434a.png)


### **실험에 대한 분석**
- 위 결과에서 보여지듯 **EB detector가 가장 높은 점수를 획득**했다.
  +  EB detector는 매우 낮은 점수를 가진 detection이 많고 false positive를 많이 생성하기 때문에 evaluation metric의 잠재적인 결함으로 점수가 높게 나온것으로 보여진다.
  +  이는 높은 재현율(Recall)과 낮은 정밀도(Precision)로 PR curve의 AUC를 효과적으로 확장했다.
  +  그러나, IOU tracker는 이러한 detections에 영향을 받지 않지만 MOTA-over-PR curve의 아래 영역(AUC)은 상당히 커지게 된다.
  +  따라서 PR curve가 **정밀도(Precision)축과 재현율(Recall)축이 있는 교차점 사이에서 완전히 정의된 경우에만 공정한 비교가 가능**할 것이다.  

  + CompACT는 다른 detector보다 PR curve의 평균 정밀도가 훨씬 우수하지만 ACF 및 R-CNN을 사용하면 더 나은 PR-MOTA 값을 달성할 수 있다.(자세한 내용은 [[17]](#5-references) 참조)
  + 왜냐하면 **CompACT의 detection은 R-CNN과 ACF의 detection보다 개수가 적고 더 정확**하기 때문이다.
  + 그러나 우리의 detector는 누락된 탐지(missing detection)를 예측할 수 없기 때문에 더 많은 detection이 있어 이점을 얻는다. 즉, detection이 적으면 좋지 않다. 
  + 특히, DETRAC evaluation script에서는 tracker를 실행하기 전에 $\sigma_l$를 사용하여 detection의 임계값을 적용한다. 
  + 일치하는 detection이 없는 경우, 이를 [8]에 따라 상당히 개선할 수 있는 방법인 $sigma < sigma_l$를 이용한 detection을 찾는것을 억제한다.   
   
  + 따라서, **track의 single missed detection은 ID switch와 false negative를 모두 야기시켜 전체 성능을 저하**시킨다. 즉, 적은 detection은 성능을 저하시킨다.
  + 반면, false positives는 보통 점수가 낮은 detections으로 구성된 짧은 track들을 생성하기 때문에 어느 정도 배제 될 수 있다.
  + 이러한 track들은 $sigma_h$및 최소 길이 $t_{min}$을 사용하여 점수가 높은 detections에 임계값을 적용함으로써 filtering한다.

- Evaluation에 기반하여, 본 논문에서 제안한 tracker는 DETRAC-Test 데이터에 대해 R-CNN detection(sigma_IOU=0.5, sigma_h=0.7, t_min=2)과 EB detection(sigma_IOU=0.5, sigma_h=0.8, t_min=2)을 이용하여 테스트가 진행한다.(6개의 최신 tracker들에 대한 비교 결과는 Table 3에 있다.)

- 본 논문에서 제안한 IOU tracker는 정확도(PR-MOTA) 및 정밀도(PR-MOTP)에 대한 전체 측정기법(metric)에서 뿐만 아니라 mostly tracked(PR-MT)및 mostly lost(PR-ML)에 대해서도 다른 방법을 상당히 능가한다.
- 특히, R-CNN detections의 경우에는 100Kfps이상의 속도가 달성되는데, 이는 0.7~390fps의 기준 방법들보다 빠른 수치이다.
- EB detector가 높은 점수의 detection 양이 많으면, 더 넓은 범위의 다양한 $sigma_l$에서 처리할 detection의 수를 크게 증가시켜 달성 가능한 fps의 속도를 감소시킨다.

training data와 overall test data 사이의 큰 성능 차이가 나는것은 해당 datasets 사이의 서로 다른  detector들의 정확도(accuracy)와 연관이 있을 것이다.  

Figure 3은 최종 PR-MOTA 및 PR-MOTP 점수에 요약된 $sigma_l$의 다른 값에 대한 MOTA 및 MOTP 점수를 보여준다.

![image](https://user-images.githubusercontent.com/71332005/224942567-e65b3893-bebe-4aeb-b0fc-fed343971ca8.png)

DETRAC 데이터셋 같은 경우에는 IOU tracker와 같은 단순한 tracking 기법이 지난 10년간의 연구에 기반한 접근들보다 더 우수한 결과를 보여준다.  

그러나, 보행자 tracking과 같은 것은 짧은 프레임에서 크기와 종횡비에 더 많은 변화가 생긴다. 이러한 변화로 인해 occlusion이 높고 프레임률이 낮으면 overlap을 계산할 때 detection를 일치시키는 성공률이 낮아 질 수 있고, 이는 보다 정교한 방법이 필요하다는 것을 나타낸다.

- 이러한 점을 고려하여, 보행자 tracking을 위한 MOT16/MOT17[11] 벤치마크에 대해 IOU tracker의 성능을 평가한다.(이 또한 최적의 파라미터를 설정하고, 테스트를 진행한다.)

![image](https://user-images.githubusercontent.com/71332005/224945590-9eada754-5184-42b0-ae4d-b4eb0b1631ae.png)

FR-CNN을 이용한 IOU tracker는 평균보다 다소 높은 성능을 달성하였으며, 보다 정확한 SDP detections을 이용할 경우 MOTA score를 크게 높일 수 있으며, 본 논문 작성 시점에서는 64개의 trackers들 중에서 13위의 성능을 보였다.
이는 보행자 tracking, 카메라 이동, 다양한 frame rates 등 보다 도전적인 상황에서도 경쟁력 있는 결과를 보여주고 있다.

추가적으로, 정적인 카메라에서 고정된 크기의 객체를 다루는 차량 tracking의 경우 높은 frame rates에서도 높은 정확도의 detections을 보였으며, 단순하면서 우수한 tracking을 달성할 수 있었다.

이러한 tracking 접근의 결과가 새로운 tracking 벤치마크의 고안을 위해 고려되길 추천한다. 

## 4. Conclusions
 본 논문에서, 우리는 간단한 수단으로 성공적인 추적이 가능하다는 것을 보여주었다. 우리가 제시한 IOU 추적기는 낮은 복잡성과 적은 계산 비용으로 최신 기법들의 성능을 크게 능가한다. 이것은 CNN 기반 접근법의 유행과 최근 객체 감지 영역의 발전으로 인하하여 가능해졌다. 일반적으로 더 높은 비디오 프레임률과 결합하여 tracking-by-detection 프레임워크에서 multi-object tracker에 대한 요구 사항이 크게 변경되었다. 우리의 간단하지만 효과적인 IOU tracker는 이러한 특성을 활용하며 새로운 조건 내에서 추적기의 설계를 반영하는 예가 될 수 있다.
 
## 5. References
\[5\] : Z. Cai, M. Saberian, and N. Vasconcelos. Learning complexity-aware cascades for deep pedestrian detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3361–3369, 2015.  
\[7\] : P. Dollar, R. Appel, S. Belongie, and P. Perona. Fast feature ´ pyramids for object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(8):1532–1545, 2014.  
\[8\] : V. Eiselein, E. Bochinski, and T. Sikora. Assessing post-detection filters for a generic pedestrian detector in a tracking-by-detection scheme. In Analysis of video and audio ”in the Wild” workshop at IEEE AVSS17, Lecce, Italy, Aug. 2017.  
\[9\] : P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan. Object detection with discriminatively trained partbased models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(9):1627–1645, 2010.  
\[10\] : R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 580–587, 2014.  
\[16\] : L. Wang, Y. Lu, H. Wang, Y. Zheng, H. Ye, and X. Xue. Evolving boxes for fast vehicle detection. Proceedings of the IEEE International Conference on Multimedia and Expo (ICME), 2017.

**Reference**  
[1. High-Speed Tracking-by-Detection Without Using Image Information, Erik Bochinski, Volker Eiselein and Thomas Sikora](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf)  
[2. \[논문읽기/2017\] High-Speed Tracking-by-Detection Without Using Image Information, neverabandon.tistory.com](https://neverabandon.tistory.com/16)


---
#### 분류성능평가지표(Classification Evaluation Metrics)
모델의 성능을 평가하기 위한 지표인 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 score등이 있다.  
이러한 평가지표를 알아보기에 앞서 혼행렬(Confusion matrix)를 알아보자.

1. 혼돈행렬(Confusion Matrix)  
모델이 예측한 결과(Predicted Label)와 실제 결과(Real Label)를 비교하는 표이다. 
![Confusion_Matrix](https://user-images.githubusercontent.com/71332005/224758716-fec85a35-d559-437d-afd2-3abdbca7a1b3.png)
- True Positive(TP) : 실제 Positive인 정답을 Positive라고 예측 (True)
- False Positive(FP) : 실제 Negative인 정답을 Positive라고 예측 (False)
- False Negative(FN) : 실제 Positive인 정답을 Negative라고 예측 (False)
- True Negative(TN) : 실제 Negative인 정답을 Negative라고 예측 (True)

2. 정밀도(Precision)  
모델이 Positive로 예측한 결과 중 실제 Positive인 결과의 비율이다. 즉, FP를 줄이는 것에 초점을 둔다.  
$$Precision=\frac{TP}{TP+FP}$$  

3. 재현율(Recall)  
실제로 Positive한 결과 중 모델이 Positive로 예측한 결과의 비율이다. 즉, FN을 줄이는데 초점을 주며, True Positive Rate(TPR)또는 민감도(Sensitivity)라고도 한다.  
$$Recall=\frac{TP}{TP+FN}$$  

4. 정확도(Accuracy)  
정확하게 예측한 데이터 개수를 총 데이터 개수로 나눈것이다.  
$$Accuracy=\frac{TP+TN}{TP+FN+FP+TN}$$ 
정확도는 실제 데이터 비율에 영향을 많이 받는 치명적인 단점이 있다. 전체 데이터 중 99%가 Posotive인 데이터가 있고, 모델이 데이터 전부를 Positive라고 예측하더라도 99%의 정확도를 나타낸다.

5. F1 Score  
앞서 정밀도와 재현율은 Trade-off 관계가 있다. 이러한 관계는 정밀도가 좋아지면 재현율이 나빠지고, 정밀도가 나빠지면 재현율이 좋아지는 관계이다.
이를 적절히 고려하여 종합적이 석분석를 하기 위한 평가이다.
$$F1 Score=\frac{2\times precision\times recall}{precision + recall}$$ 

6. ROC curve  
ROC(Receiver Operating Characteristics) 곡선은 x축을 False Positive Rate로 y축을 True Positive Rate(TPR)로 하여 생긴 곡선을 의미한다.

$$TPR(Recall)=\frac{TP}{TP+FN}$$  

$$FPR=\frac{FP}{FP+TN}$$  

다음과 같은 그래프가 나온다.  
![image](https://user-images.githubusercontent.com/71332005/224894087-fccef92f-c3eb-4eeb-a08f-27cb1f67326e.png)  

이 그래프는 True Positive Rate(TPR)와 False Positive Rate(FPR)의 값 사이의 균형을 나타낸다고 보인다.   

위와 같은 그래프를 수치로 나타내기 위해 AUC(Area Under Curve)을 사용하는데 이는 값으로 곡선 아래부분 넓이로 값을 나타낸다. 즉, AUC가 클 수록 더 좋은 분류기이다.   

다음 블로그에 아주 친절한 설명이 있다.  
[공돌이의 수학정리노트, ROC curve](https://angeloyeo.github.io/2020/08/05/ROC.html)  

7. P-R curve  
P-R curve(Precision-Recall curve)는 위에 간단히 설명한 정밀도와 재현율의 관계를 나타낸다.   
Dataset의 label 분포가 불균등 할때 P-R curve가 ROC curve에 비해 분석에 유리하다.

아래와 같은 그래프가 나온다.  
![image](https://user-images.githubusercontent.com/71332005/224896995-add6657a-b60a-4b19-aacb-591516122c27.png)

*일단 간략히 용어와 느낌을 알아보았다. 어떤 평가지표인지는 알겠지만, 막상 후에 나의 모델을 분석하기 위해 사용해야 한다면 꽤 머리가 아플것 갇다. 따라서 성능 평가지표는 따로 자세힘 정리가 필요해 보인다.*


**Reference**  
[1. Recall, Precision, F1, ROC, AUC, and everything, TechnoFob](https://technofob.com/2019/05/27/recall-precision-f1-roc-auc-and-everything/)  
[2. Classification Evaluation Metrics (분류성능지표), deeesp.gighub.io](https://deeesp.github.io/machine%20learning/Classification-Evaluation-Metrics/)

