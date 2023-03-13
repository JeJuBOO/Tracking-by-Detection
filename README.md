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

## 2. Method
- **가정**   
  + 검출기(detector)는 매 프레임마다 추적(track)할 모든 객체를 검출(detect)한다. 즉, 
  + 
