# Tracking-by-Detection
[논문 리뷰] High-Speed Tracking-by-Detection Without Using Image Information 

## Abstract
이 논문은 영상의 프레임 증가에 따라 기존 알고리즘과 견줄 수 있는 훨신 간단한 알고리즘을 제시하고 있다.

## 1. Introduction
- **객체 추적의 사용 사례** : 객체 추적은 트래픽 분석, 스포츠 또는 포렌식과 같은 분석 시스템에 정보를 제공하거나, 차량의 번호판을 인식 얼굴 인식과 같은 곳에서 사용되고 있다.
- **한계점 제시** : 특히,  tracking-by-detection systems의 일반적인 어려움 온라인으로 진행될 경우 false positive 와 missed detections가 나타날 수도 있는 제한된 성능이다. 또한 여러 객체들이 서로 교차하며 경계가 모호해질 경우 더 많은 문제들이 발생한다.
- **한계점을 해결하는 다양한 방법들을 제시하고 있다.**
- 따라서 이 논문은 [참고된 논문\[8\]][8]에서 소개된 아이디어를 기반하여 단순한 tracking 접근에 대해서 평가를 한다. 
- **논문의 활용성** : 제안하는 방법은 다른 방법들에 대한 단순한 방법으로 사용될 수 있고, 추가되는 tracking 알고리즘에 있어 기준이 될수 있다.

[8]: https://ieeexplore.ieee.org/document/8078484 

---
#### 분류성능평가지표(Classification Evaluation Metrics)
모델의 성능을 평가하기 위한 지표를 간단하게 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 score 등이 있다. 
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
----
**Reference**  
[1. Recall, Precision, F1, ROC, AUC, and everything, TechnoFob](https://technofob.com/2019/05/27/recall-precision-f1-roc-auc-and-everything/)  
[2. Classification Evaluation Metrics (분류성능지표), deeesp.gighub.io](https://deeesp.github.io/machine%20learning/Classification-Evaluation-Metrics/)

## 2. Method
- **가정**   
  + 검출기(detector)는 매 프레임마다 추적(track)할 모든 객체를 검출(detect)한다. 즉, detection에 "gap"이 아예 없거나 어이 없다.
  + 충분히 높은 프레임을 사용할 때 일반적으로 발생하는 overlap IOU가 충분히 높다고 가정한다.

$$IOU(a,b)=\frac{Area(a)\bigcap Area(b)}{Area(a)\bigcup Area(b)}$$  

- 위 두가지 가정이 모두 충족된다면, 이미지 정보 없이도 수행될 수 있다.  
- *이 논문은 특정한 임계값 $\sigma_{IOU}$를 만족할 경우 이전 프레임의 탐지 결과와 가장 높은 IOU값을 같는 탐지결과를 연관 시킴으로서 간단한 IOU Tracker를 제시한다.*

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












