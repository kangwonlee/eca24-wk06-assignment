## wk05 bisection_step

* Exercise file 실습파일 : wk05.py
* Implement one step of bisection method in function `wk05()` as follows.<br>함수 `wk05()`에 이분법의 한 단계를 구현하시오.
* Accept following arguments<br>다음과 같은 매개변수를 받아들이시오 :

argument<br>매개변수 | type<br>변수형 | description<br>설명
:-----:|:-----:|-----
`f` | `Callable[[float], float]` | the function that we want to find `x` satisfying $f(x)=0$<br>근을 찾고 싶은 함수<br>takes one float as input and returns a float<br>입력으로 실수 하나를 받아 들이고 결과값으로 실수 하나를 반환할 것임
`x_lower` | `float` | lower bound of the interval<br>구간의 하한
`x_upper` | `float` | upper bound of the interval<br>구간의 상한
`epsilon` | `float` | degree of tolerance<br>오차 허용 한계

* Return a dictionay with folliwing key-value pairs<br>다음과 같은 key-value 를 담은 `dict`를 반환하시오.
* If function `f` returns results of same signs at `x_lower` and `x_upper`, raise a `ValueError`.<br>함수 `f` 가 `x_lower` 와 `x_upper` 에서 같은 부호의 결과를 반환하면, `ValueError` 예외를 발생시키시오.

key | type of value<br>value 의 자료형 | description<br>설명
:-----:|:-----:|-----
`'x_lower'` | `float` | new upper bound<br>구간의 새로운 상한
`'x_upper'` | `float` | new lower bound<br>구간의 새로운 하한
`'found'` | `bool` | have we found it?<br>찾았나?

* Withtin the file, leave lines belong to the functions only.<br>해당 파이썬 스크립트 파일에는 해당 함수만 제출 바랍니다.
* Do not use any other modules<br>다른 모듈은 사용하지 마시오.
* Change wk05.py file only<br>wk05.py 파일	만 변경하시오.

## How to use Github web editor<br>Github 웹 편집기 사용법
* Press <kbd>.</kbd> key to start MS VS Code web editor<br><kbd>.</kbd> 키를 누르면 MS VS Code 의 Web version 이 시작됨
* Make changes to the file<br>파일을 수정
* From the left bar with the three horizontal lines at the top, (right below the magnifying glass) choose third icon, Source Control<br>왼쪽에서 줄 셋 아래 (확대경 다음) 세번째 가지치기 아이콘 선택
* Choose filename to see changes<br>변경 사항을 보려면 파일 이름 선택
* To stage changes to commit, choose + on the right side of the filename <br>수정 사항을 commit 등록 대상으로 add 추가 하려면 파일 이름의 오른쪽 + 기호 선택
* Describe the changes in the blank above<br>위 빈칸에 변경 사항 설명 입력
* Choose Commit<br>[커밋 및 푸시] 선택
* To return to the repository, use the command in the three horizontal lines<br>줄 셋 의 [리포지토리로 이동] 선택하여 저장소로 복귀
