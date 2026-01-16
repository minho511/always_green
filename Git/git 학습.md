https://abled.tistory.com/category/git?page=4

> 2026-1-15
****
## 깃 환경 설정하기

```shell
git config --global user.name "minho"
git config --global user.email "m980107@naver.com"
```
- `--global` 옵션 : 현재 컴퓨터에 있는 모든 저장소에서 같은 사용자 정보를 사용

****
## 스테이지와 버전

- **작업트리(working tree)** : 파일 수정, 저장 등의 작업을 위한 디렉토리
- **스테이지(stage, staging area)** : 버전으로 만들 파일이 대기하는 곳
- 레포지토리(repository, 저장소) : 스테이지에서 대기하고 있던 파일등르 버전으로 만들어 저장하는 곳

스테이지와 레포지토리는 깃을 초기화했을 대 만들어지는 `.git` 디렉토리 안에 숨은 파일 형태로 존재

**`.git` 안의 스테이지와 레포지토리 영역에서 버전을 만드는 과정**

- `info.txt` 파일을 수정하고 저장 -> **작업 트리**
- "버전으로 만들고 싶음" -> info.txt를 **스테이지**에 넣음 (다른 파일들도 넣음)
- 수정을 끝내고 "커밋(Commit)" 명령 -> **새로운 버전이 생성**되면서 스테이지에 대기하던 파일이 모두 **레포지토리에 저장**됨.
> 작업 트리에서 파일을 수정 -> 수정한 파일 중 버전으로 만들 파일을 스테이지에 저장 -> 스테이지에 있던 파일을 커밋 -> 레포지토리에 저장

****
## 명령어 `git init`

레포지토리를 만들고 싶은 디렉토리를 만들고, 깃을 초기화 `git init`
-> 해당 디렉토리에 있는 파일들의 버전관리를 할 수 있음

![[Pasted image 20260115192658.png]]
![[Pasted image 20260115192800.png]]

****
## 명령어 `git status`

![[Pasted image 20260115192919.png]]
1. On branch main: 현재 main 브랜치에 있음
2. No commits yet: 커핏한 파일이 없음
3. noting to commit: 커밋할 파일이 없음

 `hello.py` 생성 후 `git status`
 ![[Pasted image 20260115193213.png]]

`hello.py` 라는 untracked file이 존재함.
깃에서 한번도 버전을 관리하지 않은 파일을 의미. (스테이지, 레포지토리 둘다에 존재하지 않음)

****
## 명령어 `git add`

**생성,수정한 파일을 스테이징**

스테이징(staging): 스테이지에 파일을 추가하며 버전을 만들 준비
```
git add hello.py
```
![[Pasted image 20260115201041.png]]
untracked files 에서 changes to be committed 로 바뀜
new file -> "새로운 파일 hello.py 를 커밋하겠다."
> 스테이지에 추가되었음.

****

## 명령어 `git commit`

'커밋' 한다 -> 버전을 만든다.
```
git commit -m "create hello.py"
```
![[Pasted image 20260115201415.png]]
파일 1개가 변경되었고, 파일에 1개의 내용이 추가되었음.
> 스테이지에 있던 hello.py가 레포지토리(저장소)에추가됨.

![[Pasted image 20260115201504.png]]
스테이지에서 버전으로 넘겼기 때문에 버전으로 만들 파일이 없고 작업트리도 수정사항 없이 깨끗함.

**** 

## 명령어 `git log`

![[Pasted image 20260115201623.png]]
- 커밋을 만든사람, 만든 시간과 커밋메세지를 확인할 수 있음.

****

## 스테이징과 커밋을 한 번에 처리

커밋(commit) 명령에 -am 옵션을 사용하여 한번에 처리
> 한번이라도 커밋한 적인 파일을 다시 커밋할 때만 사용할 수 있음.

```
git commit -am "Add name"
```

![[Pasted image 20260115202124.png]]

****
## 커밋(commit) 자세히

아래와 같이 `git log` 로 얻은 로그 정보를 커밋 로그(commit log) 라고 함.
```
(base) minho@minhoui-MacBookAir test-git % git log
commit 146551e36db1d55a3edbdca08ae0207b915576c6 (**HEAD** -> **main**)
Author: minho <minho@minhoui-MacBookAir.local>
Date:   Thu Jan 15 20:20:49 2026 +0900

    Add name

commit 45aef5947e7c0798805ca46de880488880163702
Author: minho <minho@minhoui-MacBookAir.local>
Date:   Thu Jan 15 20:13:59 2026 +0900
  
    create hello.py
```

- `146551e36db1d55a3edbdca08ae0207b915576c6` : 커밋 해시(commit hash), 깃 해지(git hash), 커밋을 구별하는 아이디
- `(HEAD -> main)` : **버전이 가장 최신**이라는 표시
- `Anthor` : 누가 만든 버전인지
- `Date` : 언제 만들어진 버전인지

****

## 명령어 `git diff`

- 작업 트리에 있는 파일과 스테이지에 있는 파일을 비교
- 스테이지에 있는 파일과 레포지토리에 있는 최신 커밋을 비교
하여 수정한 파일을 커밋하기 전에 검토할 수 있음.

`hello.py` 수정전
```python
print("Hello World!")
print("Minho")
```

`hello.py` 수정후
```python
print("Hello World!")
print("Hello, Minho!")

```
(수정후 마지막에 빈 줄 있음)


> `hello.py` 파일이 수정은 됐지만 스테이징 상태가 아님.

현재 working tree의 `hello.py` 가 레포지토리에 있는 가장 최신 버전의 `hello.py` 와 어떻게 다른지 확인

![[Pasted image 20260115203356.png]]

- -print("Minho") : 해당 줄이 삭제되었고,
- +print("Hello, Minho!") : 해당 줄이 추가되었음.
- index ... : 파일의 고유 식별자(ID) 와 권한
	- `bbaf50c..8869eba` : 파일 내용의 해시(Hash)값
	  git은 파일 이름을 보지 않고, 파일의 '내용'을 암호화된 코드로 변환하여 관리.
		- `bbaf50c` : 변경 전 파일 내용의 ID
		- `8869eba` : 변경 후 파일 내용의 ID
	- `100644` : 파일의 모드(권한)
		- `100` : 일반 파일(Regular file)임을 의미
		- `644`: 파일 권한이 `rw-r--r--` 인 일반적인 텍스트 파일
- `@@ -1,2 +1,3 @@`: 변경이 일어난 위치와 라인 수에 대한 요약 -> `Hunk Header`
	- `-1,2` (변경 전 파일 정보): (-)변경 전 파일, 1번째 줄부터, 총 2줄을 해당 문맥으로 잡고 있음
	- `+1,3` (변경 후 파일 정보): (+)변경 후 파일, 1번째 줄부터, 총 3줄이 되었음
	
****

>2026-1-16