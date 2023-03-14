# Tracking-by-Detection
\[논문 리뷰\]  
[**High-Speed Tracking-by-Detection Without Using Image Information**](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf)  
Erik Bochinski, Volker Eiselein and Thomas Sikora  
Communication System Group, Technische Universitat Berlin  
Einsteinufer 17, 10587 Berlin

## Abstract
이 논문은 영상의 프레임 증가에 따라 기존 알고리즘과 견줄 수 있는 훨신 간단한 알고리즘을 제시하고 있다.

## 1. Introduction
- **객체 추적의 사용 사례** : 객체 추적은 트래픽 분석, 스포츠 또는 포렌식과 같은 분석 시스템에 정보를 제공하거나, 차량의 번호판을 인식 얼굴 인식과 같은 곳에서 사용되고 있다.
- **한계점 제시** : 특히,  tracking-by-detection systems의 일반적인 어려움 온라인으로 진행될 경우 false positive 와 missed detections가 나타날 수도 있는 제한된 성능이다. 또한 여러 객체들이 서로 교차하며 경계가 모호해질 경우 더 많은 문제들이 발생한다.
- **한계점을 해결하는 다양한 방법들을 제시하고 있다.**
- 따라서 이 논문은 [참고된 논문\[8\]](#5-references)에서 소개된 아이디어를 기반하여 단순한 tracking 접근에 대해서 평가를 한다. 
- **논문의 활용성** : 제안하는 방법은 다른 방법들에 대한 단순한 방법으로 사용될 수 있고, 추가되는 tracking 알고리즘에 있어 기준이 될수 있다.
![image](https://user-images.githubusercontent.com/71332005/224903580-f4a7b85d-3391-4bc6-8a05-57a4e9c51d8d.png)

## 2. Method
- **가정**   
  + 검출기(detector)는 매 프레임마다 추적(track)할 모든 객체를 검출(detect)한다. 즉, detection에 "gap"이 아예 없거나 어이 없다.
  + 충분히 높은 프레임을 사용할 때 일반적으로 발생하는 overlap IOU가 충분히 높다고 가정한다.

$$IOU(a,b)=\frac{Area(a)\bigcap Area(b)}{Area(a)\bigcup Area(b)}$$  

- 위 두가지 가정이 모두 충족된다면, 이미지 정보 없이도 수행될 수 있다.  
- *이 논문은 특정한 임계값* $\sigma_{IOU}$ *를 만족할 경우 이전 프레임의 탐지 결과와 가장 높은 IOU값을 같는 탐지결과를 연관 시킴으로서 간단한 IOU Tracker를 제시한다.*

- **성능향상**
  + $t_min$보다 길이가 짧거나 $\sigma_h$이상의 점수를 가진 detection이 하나도 없는 track을 제거함으로 성능을 향상시킬 수 있다.(짧은 track들은 일반적으로 false positive를 기반으로 하고 출력에 혼란(clutter)을 추가하기 때문에)
  + track에 적어도 하나 이상의 높은 score의 detection을 갖도록 요구하면, track의 완성도를 위한 낮은 score의 detection에 이점을 가진다.

- 논문에서 제안한 **알고리즘 1**이다. 
![Algirithm_1](https://user-images.githubusercontent.com/71332005/224783065-8a86d68e-d903-4abd-a078-f17df4b41ee3.png)
  + $D_f$: frame f에서의 detection
  + $d_j$: 해당 frame에서의 j번째 detection
  + $T_a$: 활성화된(actived) track
  + $T_f$: 완료된(finished) track
  + $F$: frame 수

- 알고리즘의 5번 라인에서는 가장 잘 일치하지만 할당되지 않은 detection만 track에 추가할 후보가 된다.
  + 이는 반드시 detection $D_f$와 Track $T_a$사이의 연관성으로 이어지지 않지만, 해당 프레임에서 모든 IOU들의 합을 최대화 하는 Hungatian algorithm응 적용함 으로 해결 될 수 있다.
  + 일반적인 $\sigma_{IOU}$는 검출기의 비-최대 억제(non-maxima suppression)를 위한 IOU 임계값과 같은 범위에 있으므로 가장 높은 IOU를 선택하는 것은 합리적인 huristic이다.

- 알고리즘의 이점
  + 제안한 방법의 복잡도는 다른 최신의 tracker들과 비교 결과 매우 낮다.
  + 프레임의 시각적 정보는 사용되지 않으므로 감지 수준에 대한 간단한 필터링 방법으로도 볼 수 있다.
  + tracker가 최신 검출기와 함께 온라인으로 사용되는 경우, 최신 검출기에 비해 tracker의 계산 비용은 무시할 수 있는 수준이다.
  + tracker를 단독으로 수행해도 100K의 fps를 초과하는 프레임율을 쉽게 달성할 수 있다.
  + tracker의 속도적 이점은 출력을 이미지나 움직임 정보를 사용하여 연결할 수 있는 tracklets으로 고려함으로 tracking 구성 요소를 추가할 수 있다는 점에 유의한다. 

## 3. Experiments  
- **DETRAC dataset를 사용하여 제안된 추적기의 성능을 조사했다.** DETRAC dataset는 차량 검출 및 추적을 목적으로 10시간 이상의 영상으로 구성된 데이터셋이다.
- **detector는 CompACT[[5]](#5-references), R-CNN[[10]](#5-references), ACF[[7]](#5-references), VGG16 1-3-5 모델을 이용한 Evolving Boxes detector(EB)[[16]](#5-references)등을 사용한다.**
- DPM[[9]](#5-references) 는 너무 부정확하기 때문에 제안한 tracker에 적합하지 않아 결과에 포함하지 않았다.

- [**분류성능평가지표**](#분류성능평가지표classification-evaluation-metrics)  
  + UA-DETRAC evaluation protocol을 사용하여 평사를 수행했다.
  + 이는 P-R 곡선을 계산하기 위해, tracking에 서로 다른 detection 점수의 임계값인 $\sigma_l$을 여러번 적용한다.
  + 이 곡선을 통해 일반적인 CLEAR MOT matrix[[14]](#5-references)이 계산되며, 최종적인 점수는 AUC로 구성되고 모든 detector의 임계값 $\sigma_l$에 대한 성능을 고려한다.(자세한 내용 참고[[17]](#5-references))
  + 이는 임계값 $\sigma_h$ 설정에 영향을 미치지 않고 낮은 점수의 detection를 사용하는 것에 영향을 미친다. 
  + 일반적으로 [8]에 따라 점수가 낮은 detection의 수가 많을수록 접근 방식에 대한 추적 성능이 향상된다고 가정할 수 있다.

- 구현을 오직 파이썬으로 진행했고, 어떠한 성능 최적화도 수행하지 않았다.  

- **Parameter Search**

![Table 1](https://user-images.githubusercontent.com/71332005/224924926-3ed701de-40b4-4a2f-8ba3-baf835752c92.png)

![Table 2](https://user-images.githubusercontent.com/71332005/224920606-a9805588-4113-4020-aa00-d5f0c50b6ec3.png)

![Figure 2](https://user-images.githubusercontent.com/71332005/224920837-ea2636b1-105c-4ec0-b8d2-3f9288749b2d.png)

- $\sigma_{IOU}$, $\sigma_h$, $t_{min}$에 대한 최적의 parameters는 각 detector에 대한 훈련 데이터셋에 대해 grid search를 수행해 결정했다.
- 모든 detection 점수들은 $\sigma\in \[0.0;1.0\]$로 정규화 되었으나, 여전히 detector마다 다르게 분포되어있다. 따라서, $\sigma_h$의 범위도 다르게 선택되어야 한다.

- Table 2.의 범위 내의 3가지 parameters의 모든 조합을 평가하기 위해 각 detector 당 64번 실행시킨 결과를 Figure 2로 나타냈다. UA-DETRAC 챌린지의 기본 측정기법(metric)인 PR-MOTA metric으로 최상의 구성을 선택한다.

-각 detector에 대한 최상의 결과와 구성을 Table 3에서 볼 수있다.
![Table 3](https://user-images.githubusercontent.com/71332005/224925563-1adb1ebf-f21b-4edd-b5d2-188f2ce1434a.png)


- **위 Table 3결과에 대한 분석**
  +  위 결과에서 보여지듯 EB detector가 가장 높은 점수를 획득했다.
  +  EB detector는 매우 낮은 점수를 가진 detection으로 많은 false positive를 생성함으로, 평가 metric의 잠재적인 결함으로 점수가 높게 나온것으로 보여진다.
  +  이는 높은 재현율(Recall)과 낮은 정밀도(Precision)으로 PR curve를 효과적으로 확장했다.
  +  그러나, IOU tracker는 이러한 detections에 영향을 받지 않지만 MOTA-over-PR curve의 아래 영역(AUC)은 상당히 커지게 된다.
  +  따라서 PR curve가 정밀도(Precision) 축과 재현율(Recall) 축이 있는 교차점 사이에서 완전히 정의된 경우에만 공정한 비교가 가능할 것이다.  

  + CompACT는 다른 기준 검출보다 PR 곡선의 평균 정밀도(자세한 내용은 [17] 참조)가 훨씬 우수하지만 ACF 및 R-CNN을 사용하면 더 나은 PR-MOTA 값을 달성할 수 있다. 

























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
모델의 성능을 평가하기 위한 지표인 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 score , 이 있다. 
이러한 평가치표를 알아보기에 앞서 혼동 행렬(Confusion matrix)를 알아보자.

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
정확도는 실제 데이터 비율에 영향을 많이 받는 치명적인 단점이 있다. 전체 데이터 중 99%가 Posotive인 데이터가 있다면, 모델이 데이터 전부를 Positive라고 예측하더라도 99%의 정확도를 나타낸다.

5. F1 Score
앞서 정밀도와 재현율은 Trade-off 관계가 있다. 이러한 관계는 정밀도가 좋아지면 재현율이 감소하고, 정밀도가 나빠지면 재현율이 좋아지는 관계이다.
이를 적절히 고려하여 종합적이 평가를 하기 위한 평가이다.
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

