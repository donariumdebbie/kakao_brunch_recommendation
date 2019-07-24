# Brunch Article Recommendation
아래 코드와 같이 데이터를 넣어주세요.(데이터셋이 `./res` 이외에 위치한 경우는 `config.py`를 수정하세요)
```bash
$> tree -d
.
├── res
│   ├── contents
│   ├── predict
│   └── read
└── tmp
```

아래와 같이 실행하면 제출용 파일인 recommend.txt가 나옵니다.
```bash
python recommendmarch.py
```


여러 모델을 시도해보고 싶었지만 
read 정보가 없는 기간의 user가 팔로잉하는 작가의 글, 최근 읽은 글의 작가들의 새로운 글, 최근 읽은 매거진의 새로운 글, 팔로잉하는 작가들의 좀더 오래된 글, 인기있던 most popular 글 중 팔로잉하는 글, 그 외 정보가 없는 경우 most popular로 나머지를 추천하는 로직입니다. 
