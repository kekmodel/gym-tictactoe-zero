# gym-tictactoe
![tictactoe](./img/Tic_Tac_Toe.gif)
----------------------------
### <일반인이 도전해보는 강화학습>
## 첫번째 프로젝트: gym 기반으로 틱택토 환경 만들어 보기
gym을 설치한 후 실행하면 돌아감! 파이썬 3.5로 만들었음!

gym폴더가 gym-tictactoe폴더가 있는 곳에 있어야 함!


https://github.com/openai/gym

다음 프로젝트: 내가 만든 환경에서 강화학습 에이전트 구현

수준: 코딩의 코자도 모르는 코린이 (파이썬 공부한 순수시간 24시간 정도에서 시작)

그래서 삽질의 삽질의 삽질의 연속...

혼자 삽질하면서 쥐똥만큼 알게된 내용 나름 정리 (확실한 정보가 아님;;)

!!코드에 더 자세히 설명 됨!!


### gym 기본 임포트 모듈

	import logging  로그 생성용인듯 사용법 아직 모름
	import gym
	from gym import spaces 공간 정의 도와줌
	from gym.utils import seeding 시드생성에 사용
	import numpy as np  C의 배열을 사용하게 해줌 우리의 꿈과 희망

환경 클래스는 gym.Env를 상속받아 구성함
	
	class myEnv(gym.Env):
		metadata = {'render.modes': ['human', 'rgb_array'], 
				'video.frames_per_second': 60}
		
메타데이타는 render()의 리턴 타입 구분? 사람 눈으로 볼거고 rgb값 60프레임으로 잡음
어떻게 작용되는진 모르겠음

### 필수 정의 메소드들

\_\_init\_\_(self)

	클래스 초기화 메소드
	멤버로 observation_space, action_space 정의
	정의방법은 from gym import spaces하여 spaces의 Disccrete로 정수타입을 Box로는 여러차원의 실수타입을 정의할때 사용함
	그러면 .sample()로 랜덤값 반환 가능
	_seed() 호출
	state, done, viewer 멤버 초기화하기


\_seed(self, seed=None)


	self.np_random, seed = seeding.np_random(seed)
        return [seed]
시드 생성 메소드 같은데 보통 이렇게 정의함;; 정확한 메커니즘은 모름.
np디폴트 시드를 개선한 것 같음. 인스턴스.np_random. 사용도 가능.



\_reset(self)
		
	에피소드가 끝나면 호출해야 하는 메소드, state 반환!
	reset()으로 외부에서 호출가능
	!!state 초기화 실제 상태는 여기서 만듦!!
	done 초기화
	return self.state 

\_step(self, action)

	환경의 핵심이 되는 메소드
	에이전트가 step()으로 호출
	액션에 대한 상태변화, 보상, 진행상태, 추가정보를 정함
	
	action_space의 값중 하나를 인풋으로 하여
	(다음 state, reward, done, info) 튜플 리턴
	
	에피소드가 끝나면 reset() 어떻게 할지 생각해야함
	reset()은 외부에서 컨트롤 가능

### 필수는 아니나 알아두면 좋은 메소드
\_render(self, mode='human', close=False)


상태를 그리는 함수 에이전트에서 render()로 호출함
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
이걸 로 시작하는데 없어도 될 것 같아서 없애면 에러남 필수인 듯!

렌더 호출 시 본론부분

	if self.viewer is None:
	...
	아래에 self.viewer 정의


rendering 모듈

	from gym.envs.classic_control import rendering
	간단한 렌더링 지원해줌 pyglet기반인듯

뷰어 정의

	self.viewer = rendering.Viewer(screen_width, screen_height)

줄 긋기 및 선색깔 정의 (x, y) 좌표기반, (r, g, b)


	self.line_1 = rendering.Line((0, 100), (300, 100))
            self.line_1.set_color(0, 0, 0)

뷰어에 붙이기

	.add_geom 메소드
	self.viewer.add_geom(객체)
	

이미지 객체 생성

	.Image("파일위치", 가로, 세로 크기)
	self.image_O1 = rendering.Image("img/O.png", 96, 96)

위치 지정하는 방법

	위치컨트롤 객체 생성-> 대상객체에 붙임 -> 대상을 뷰어에 붙임

위치컨트롤 객체	

	.Transform((x좌표, y좌표))
	self.trans_O1 = rendering.Transform()
	대상이 뷰어에 붙기 전까진 뷰어에 반영안됨

위치바꿀 대상객체에 붙이기

	대상객체.add_attr(트랜스폼객체)

뷰어에 대상객체 붙이기

	마찬가지로 self.viewer.add_geom(대상객체)


### 코딩일기
클래스 
  
	상속방법은 class 클래스명(슈퍼):
	필수 구성 함수는 def __init__(self, 인풋들) 
	기본변수기능: 멤버 (보통 __init__메소드 내부에 self.이름으로 선언)
				(메소드 내에서 만들기도 함)
	내부함수기능: 메소드 (행동을 담당)
	self는 클래스에 찍혀나온 인스턴스 자신을 생각하면 됨
	클래스 내부 함수와 외부 함수의 구분:
		내부 멤버들이 굳이 필요하지 않은 함수는 외부에 정의해서 인스턴스의 접근을 제한하는 것이 좋음(맞나?)
	메소드를 _이름 으로 정의 했는데
	인스턴스에서 언더바 없는 이름으로 접근 가능한 경우도 있음(API?)

 numpy
  
	같은 타입만 들어갈 수 있음!
	np.zeros((행수, 열수), '타입') 으로 초기화!
	np.where(배열 == 1)
		배열 값 중 1인 곳의 인덱스를 반환
	np.count_nonzero(배열)
		배열의 값중 0이 아닌 개수 반환
	배열.sum()
		배열 값의 총합
	지원되는 거 겁나 많음
	
 삽질
 
	함수의 리턴 타입을 신경쓰자
	for, if 문은 탈출조건이 없을 시 아래로 흐름?
	스레드 공부 필요함

 리스트
 
	[]로 선언 
	여러타입이 들어갈 수 있고
	추가 삭제 용이함
	numpy와 찰떡 궁합
	컴프리핸션 좋은 거 많음
	[ x for x in range(3)] if else도 됨;;
	=로 연결하면 참조임. 복사하고싶으면 리스트.copy()

 lambda
 
    	익명 함수
	간단한 함수를 바로 쓸 수 있게 함. 개꿀.
	lambda 인풋변수명: 변수규칙
	하면 아웃풋 반환함!	

 튜플
  
	()로 선언
	추가 안되는 리스트로 생각


 딕셔너리
 
	가장 쓸모가 많았던 놈
	{키: 값, ...} 으로 선언
	dict.keys() : 키값반환
	dict.values() :값반환
	dict.items() : 둘다반환 타입은? 공부요망
	dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8], state[1]))
	zip(키리스트, 값리스트) 각각에 맞게 묶어줌 개꿀
	dict로 머리 좀 굴리면 웬만한 문제 다 해결

 언팩킹
 
 
	*[] 타입을 벗어던지고 내부값으로 반환
	**{} 별두개는 키와 내부값. 반환 순서는? 공부요망
	*{} 한번만 하면 키만 반환
	함수 인풋 표현에도 쓰는데 솔직히 이해 안감

쓸만한 내부 함수들


	map(함수, 이터)
		순서대로 함수에 넣고 순서대로 아웃풋 반환함. 타입은 원소자체 타입으로 반환됨.
	filter(규칙, 이터)
		이터의 원소 중 규칙에 맞는 원소 순서대로 반환
		이걸 map으로하면 bool값이 아웃풋으로 나옴
