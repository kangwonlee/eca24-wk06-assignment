wk06 vibration_response_fitting

# 🚀 SciPy Optimization: Vibration Response Fitting<br>🚀 SciPy 최적화 : 진동계 응답 곡선적합

* This assignment practices fitting experimental mechanical vibration response data with an exponentially decaying sinusoidal function using `scipy.optimize`. Please <kbd>implement the cost function</kbd> to calculating the cost.<br>`scipy.optimize` 기능을 이용하여 진동계 응답 데이터를 지수적으로 진폭이 감소하는 사인 함수로 곡선적합하기 위한 <kbd>비용 함수를 작성</kbd>하시오.

## Problem Description<br>문제 설명

* We can possibly carry out a mechanical vibration experiment as follows.<br>기계 진동 실험은 다음과 같이 수행할 수도 있다.
* Suspend a spring vertically<br>스프링을 수직으로 설치한다.
* Attach a mass to lower end of the spring<br>질량을 스프링 하단에 부착한다.
* Extend the spring by a fixed length and then release.<br>스프링을 일정 길이로 늘어나도록 한 다음 놓는다.
* Measure the position of the mass over time.<br>질량의 위치를 시간에 따라 측정한다.
* The response is usually a decaying sinusoidal function due to the damping effect of the spring.<br>해당 응답은 대체로 스프링의 감쇠 효과로 인해 진폭이 지수적으로 감소하는 사인 함수이다.
* One of the goals of the experiment is to find the best-fit function of the following form that matches the experimental data points (time, acceleration):<br>실험의 목적 가운데 하나는 다음 형태의 곡선 함수를 실험 데이터 (시간, 위치)와 가장 유사하게 만드는 매개변수를 찾는 것이다.

$$ x(t) = A \cdot e^{-\zeta \omega t} \cdot sin\left(\left(\omega \sqrt{1 - \zeta^2}\right) t + \phi \right) + DC$$

symbol<br>기호 | unit<br>단위 | Description<br>설명
:---:|:---:|---
x(t) | m | position of the mass at time t<br>시간 t 에서의 질량의 위치
A | m | Initial amplitude<br>초기 진폭
$\zeta$ | $\cdot$ | Damping ratio<br>감쇠비
$\omega$ | rad/sec | Undamped angular frequency<br> 비감쇠 각 주파수
$\phi$ | rad | Phase shift<br>위상 변화
DC | m | DC component of the measured data<br>직류 성분


## Instructions<br>지시사항

* Modify `exercise.py` file only<br>`exercise.py` 파일만 변경하시오.
* Write two python functions.<br>파이썬 함수 두개를 작성하시오.
    * Function `wk06_cost()` will calculate and return the cost function.<br>함수 `wk06_cost()` 은(는) 비용 함수를 계산하여 반환하시오.
    * Function `wk06_curve()` will evaluate and return the a(t) above.<br>함수 `wk06_curve()` 는 위 $x(t)$ 를 계산하여 반환하시오.
* Within the `exercise.py` file, every code line should be inside a function.<br>`exercise.py` 파일 내의 모든 코드 행은 함수 내에 있어야 함.
* Please use `numpy` and `scipy`.<br>`numpy` 와 `scipy` 를 사용하시오.
* To visualize (for example on the Colab), use `matplotlib`.<br>시각화를 위해 (예를 들어 Colab 상에서) `matplotlib` 를 사용하시오.
* Do not use global variables. Use function arguments and return values.<br>전역 변수를 사용하지 마시오. 함수의 인자와 반환 값을 사용하시오.
* See `sample.py` for possible use of the function(s).<br>해당 함수 사용 예는 `sample.py` 를 참고하시오.

## `wk06_cost()` Function:

* Function `wk06_cost()` will take the following arguments (see table below).<br>함수 `wk06_cost()` 는 다음과 같은 인자를 받아들인다 (아래 표 참조).

| Argument<br>인자 | Data Type<br>자료형 | Description<br>설명 
|:---:|:---:|---
| `param` | numpy array | Array of current parameter values (A, zeta, w, phi, offset)<br>현재 매개변수 값 배열 (A, zeta, w, phi, offset)
| `t` | numpy array | time samples of the experiment<br>각 값을 측정한 시간
| `x` | numpy array | measured data<br>일정 시간 간격으로 측정된 데이터

* Please remeber that the `param` array contains the following parameters in this order:<br>`param` 배열은 순서대로 다음 매개변수를 포함한다:

    * `A` - Initial amplitude<br>초기 진폭
    * `zeta` - Damping ratio<br>감쇠비
    * `w` - Undamped angular frequency<br>비감쇠 각 주파수
    * `phi` - Phase shift<br>위상 변화
    * `offset` - DC component of the measured data<br>측정된 데이터의 직류 성분

* The function should return the Root Mean Squared Error (RMS Error) between the fitted function and the data (see table below).<br>해당 함수는 곡선 함수와 데이터 사이의 제곱 평균 제곱근 오차 (RMS 오차)를 반환한다(아래 표 참조).

| Return Value<br>반환값 | Data Type<br>자료형 | Description<br>설명
|:---:|:---:|---|
| `rms_error` | `float` | The calculated RMS Error<br>제곱 평균 제곱근 오차

## `wk06_curve()` Function:

* Function `wk06_curve()` will take the following arguments (see table below).<br>함수 `wk06_curve()` 는 다음과 같은 인자를 사용한다(아래 표 참조).

| Argument<br>인자 | Data Type<br>자료형 | Description<br>설명 
|:---:|:---:|---
| `A`      | `float` | Initial amplitude<br>초기 진폭
| `zeta`   | `float` | Damping ratio<br>감쇠비
| `w`      | `float` | Undamped angular frequency<br>비감쇠 각 주파수
| `phi`    | `float` | Phase shift<br>위상 변화
| `offset` | `float` | DC component of the measured data<br>직류 성분
| `t` | numpy array | time samples of the experiment<br>각 값을 측정한 시간

* The function should return estimated the x(t) above.<br>해당 함수는 x(t) 함수 추정값을 반환한다(아래 표 참조).

| Return Value<br>반환값 | Data Type<br>자료형 | Description<br>설명
|:---:|:---:|---|
| `x_hat` | numpy array | Estimated x(t) values at each t steps<br>t 각 단계에서 x(t) 추정값

* Please check the GitHub Actions results. If "Autograding" and "Check message" results are different, try re-running the failed one.<br>Github Actions 결과를 확인하시오. "Autograding" 과 "Check message" 결과가 다르면, 통과되지 않은 시험을 재시도 해보시오.
* "Check message" results may include artifacts. If yes, please download and check the results.<br>"Check message" 결과에는 다운로드용 artifact 가 있을 수 있음. 있는 경우 받아 보고 결과를 확인하시오.

### Let's optimize! 🏆<br>최적 함수를 찾아봅시다! 🏆

``From here is common to all assignments.``

## How to use Github web editor<br>Github 웹 편집기 사용법
* Press <kbd>.</kbd> key to start MS VS Code web editor<br><kbd>.</kbd> 키를 누르면 MS VS Code 의 Web version 이 시작됨
* Make changes to the file<br>파일을 수정
* From the left bar with the three horizontal lines at the top, (right below the magnifying glass) choose third icon, Source Control<br>왼쪽에서 줄 셋 아래 (확대경 다음) 세번째 가지치기 아이콘 선택
* Choose filename to see changes<br>변경 사항을 보려면 파일 이름 선택
* To stage changes to commit, choose + on the right side of the filename <br>수정 사항을 commit 등록 대상으로 add 추가 하려면 파일 이름의 오른쪽 + 기호 선택
* Describe the changes in the blank above<br>위 빈칸에 변경 사항 설명 입력
* Choose Commit<br>[커밋 및 푸시] 선택
* To return to the repository, use the command in the three horizontal lines<br>줄 셋 의 [리포지토리로 이동] 선택하여 저장소로 복귀

## Notes

* This assignment was developed with the assistance of an AI language model for creative inspiration and guidance, demonstrating the potential of AI as a tool to enhance learning and problem-solving.  Students are encouraged to use AI responsibly and ethically, always prioritizing their own understanding and critical thinking.

## NOTICE REGARDING STUDENT SUBMISSIONS<br>제출 결과물에 대한 알림

* Your submissions for this assignment may be used for various educational purposes. These purposes may include developing and improving educational tools, research, creating test cases, and training datasets.<br>제출 결과물은 다양한 교육 목적으로 사용될 수 있을 밝혀둡니다. (교육 도구 개발 및 개선, 연구, 테스트 케이스 및 교육용 데이터셋 생성 등).

* The submissions will be anonymized and used solely for educational or research purposes. No personally identifiable information will be shared.<br>제출된 결과물은 익명화되어 교육 및 연구 목적으로만 사용되며, 개인 식별 정보는 공유되지 않을 것입니다.

* If you do not wish to have your submission used for any of these purposes, please inform the instructor before the assignment deadline.<br>위와 같은 목적으로 사용되기 원하지 않는 경우, 과제 마감일 전에 강사에게 알려주기 바랍니다.

``Until here is common to all assignments.``
