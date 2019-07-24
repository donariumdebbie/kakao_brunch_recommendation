# -*- coding: utf-8 -*-
import os
try:
    import cPickle
except ImportError:
    import pickle as cPickle

import fire
import tqdm

import config as conf
from util import iterate_data_files
import pandas as pd
from datetime import timedelta, datetime


class RecommendMarch(object):
    topn = 100

    def __init__(self, from_dtm, to_dtm, tmp_dir='./tmp/'):
        self.from_dtm = str(from_dtm)
        self.to_dtm = str(to_dtm)
        self.tmp_dir = tmp_dir

    def _get_model_path(self):
        model_path = os.path.join(self.tmp_dir, f'mp.model.{self.from_dtm}.{self.to_dtm}.recent')

        return model_path

    def _build_popular_model(self):
        model_path = self._get_model_path()
        if os.path.isfile(model_path):
            return

        freq = {}
        print('building model..')
        for path, _ in tqdm.tqdm(iterate_data_files(self.from_dtm, self.to_dtm), # 이 기간 사이에 많이 읽힌 글을 추천
                                 mininterval=1):
            for line in open(path):
                seen = line.strip().split()[1:] # read에서 읽어온 한 줄 한줄을 쪼개고 뒷부분이 읽은글리스트
                for s in seen: # 줄마다 읽은글리스트의 읽은글들에 대해
                    freq[s] = freq.get(s, 0) + 1 #  freq이라는 dictionary의 읽은글의 frequency값을 가져오고 없으면 0을 가져옴(default 0),그리고 1더해줌 
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True) # freq을 정렬할 때 reverse로..
        open(model_path, 'wb').write(cPickle.dumps(freq, 2))
        print('model built')

    def _get_popular_model(self):
        model_path = self._get_model_path()
        self._build_popular_model()
        result = cPickle.load(open(model_path, 'rb'))
        return result


    def _metadf_select_dates(self, df_meta, startdate = '2019-03-01', enddate = '2019-03-14'):
        # df_meta = df_meta.copy()
        df_meta['reg_datetime'] = df_meta['reg_ts'].apply(lambda x : datetime.fromtimestamp(x/1000.0))
        df_meta.loc[df_meta['reg_datetime'] == df_meta['reg_datetime'].min(), 'reg_datetime'] = datetime(2090, 12, 31)
        df_meta['reg_dt'] = df_meta['reg_datetime'].dt.date
        df_meta['reg_dt'] = pd.to_datetime(df_meta['reg_dt'])
        df_meta_filtered = df_meta.loc[df_meta['reg_dt'].isin(pd.date_range(startdate, enddate))] # 날짜 범위 내로 걸러진 데이터프레임
        return df_meta_filtered

    def _filter_df_articles(self, df_tofilter, authorlist):
        df_filtered = df_tofilter[df_tofilter['user_id'].isin(authorlist)]
        return df_filtered.index.tolist(), df_filtered['user_id'].tolist() # 글식별자랑 작가아이디만 리턴

    def _filter_df_articles_key(self, df_tofilter, comparelist, key):

        df_filtered = df_tofilter[df_tofilter[key].isin(comparelist)]


        return df_filtered.index.tolist(), df_filtered[key].tolist() 

    def _get_seens(self, users):
        set_users = set(users)
        seens = {}
        for path, _ in tqdm.tqdm(iterate_data_files(self.from_dtm, self.to_dtm),
                                 mininterval=1):
            for line in open(path):
                tokens = line.strip().split()
                userid, seen = tokens[0], tokens[1:]
                if userid not in set_users:
                    continue
                seens[userid] = seen # seens라는 딕셔너리에 유저(key)가 그 기간에 읽은 글의 목록을 저장해둠 
        return seens

    def _get_following(self, user_id, users_df):  # user가 following 하는 작가들을 저장
        if user_id not in users_df.index:
            print('no following authors')
            user_following_authors = []
            return user_following_authors

        user_following_authors = users_df['following_list'].loc[user_id]
        return user_following_authors
    
    def _get_seen_keywords(self, seen_articles, metadata_df): # user가 읽은 글들의 키워드들을 저장 
        user_seen_keywords = set([])
        for article in seen_articles:
            if article not in metadata_df.index:
                continue
            keywords = metadata_df['keyword_list'].loc[article]
            user_seen_keywords.update(keywords) # 리스트일때 업데이트, set일때 add
        return user_seen_keywords

    def _get_seen_authors(self, seen_articles, metadata_df): # user가 읽은 글들의 작가들을 저장
        user_seen_authors = set([])
        for article in seen_articles:
            if article not in metadata_df.index:
                continue
            author = metadata_df['user_id'].loc[article]
            user_seen_authors.add(author)
        return user_seen_authors

    def _get_seen_magazines(self, seen_articles, metadata_df): # user가 읽은 글들의 매거진들을 저장
        user_seen_magazines = set([])
        for article in seen_articles:
            if article not in metadata_df.index:
                continue
            magazine = metadata_df['magazine_id'].loc[article]
            user_seen_magazines.add(magazine)
        return user_seen_magazines

    def _get_author(self, article, metadata_df):
        if article in metadata_df.index:
            return metadata_df['user_id'].loc[article]
    
    def _get_magazine(self, article, metadata_df):
        return metadata_df['magazine_id'].loc[article]
    
    def _get_keywords(self, article, metadata_df):
        return metadata_df['keyword_list'].loc[article]


    def _score_records(self, recs_partial, user_df, metadata_df, threshold_score, 
    condition_following_author, condition_keyword, condition_seen_author, condition_seen_magazine):
        recs_filtered = []
        for article in recs_partial:
            # 팔로잉하는 작가이면 + 3
            # 글에 봣던 키워드가 있으면 + 1 ## TODO 개선안으로 키워드 갯수별로 점수 다르게 
            # 봤던 글의 작가이면 + 2 
            # 봤던 매거진의 글이면 +2  ## TODO 연속되게 본 글의 매거진인지 체크 
            score = 0
            this_author = self._get_author(article, metadata_df)
            this_magazine = self._get_magazine(article, metadata_df)

            if this_author in condition_following_author:
                score += 3
            if bool(set(self._get_keywords(article, metadata_df)).intersection(set(condition_keyword))):
                score += 1
            if this_author in condition_seen_author:
                score += 2
            if this_magazine !=0 and this_magazine in condition_seen_magazine:
                score += 2
            
            if score >= threshold_score:
                recs_filtered.append(article)

        return recs_filtered


################################################################################
    def recommend(self, userlist_path, out_path):
        directory = conf.data_root

        #TODO : 2/22-3/14 글 일만개 가져오기 df로
        metadata_df = pd.read_json(os.path.join(directory, 'metadata.json'), lines = True)
        metadata_df = metadata_df.set_index('id') # 메타데이타 데이터프레임의 글식별자를 index(딕셔너리의 키)로 만들어줌

        meta_recent_df = self._metadf_select_dates(metadata_df, '2019-02-22', '2019-03-14') # 제출용 기간 
        meta_older_df = self._metadf_select_dates(metadata_df, '2019-01-01', '2019-02-22')

        users_df = pd.read_json(os.path.join(directory, 'users.json'), lines=True)
        users_df = users_df.set_index('id') # 유저 데이터프레임의 사용자식별자를 index(딕셔너리의 키)로 만들어줌

        most_popular = self._get_popular_model() # 위에서 만든 frequency dictionary 모델. 포스트별로 몇 번 봤는지 카운팅해온것. 
        most_popular = [a for a, _ in most_popular]
        most_popular_df = meta_recent_df.loc[most_popular]

        with open(out_path, 'w') as fout: # 파일 작성을 한다
            users = [user.strip() for user in open(userlist_path)] # dev.users 아니면 test.users파일. 한줄씩 user불러옴. 

            seens = self._get_seens(users)

            for user in tqdm.tqdm(users):
                rec100 = []
                checkrec100 = set(rec100)
                seen_articles = set(seens.get(user, [])) # 한 유저가 본 아티클 전체목록 ### TODO 여기가 좀 오래걸림. 근데 베이스라인에 있던부분. 

                # TODO : 이 유저가 팔로잉하는 작가 목록 가져오기 
                user_following_authors = set(self._get_following(user, users_df))

                # 추천할 때 마다 중복체크

                user_seen_authors = set(self._get_seen_authors(seen_articles, metadata_df))
                user_seen_magazines = set(self._get_seen_magazines(seen_articles, metadata_df))

                recent_following_articles, _ = self._filter_df_articles(meta_recent_df, user_following_authors)
                recent_seen_authors_articles, _ = self._filter_df_articles(meta_recent_df, user_seen_authors)
                recent_seen_magazine_articles, _= self._filter_df_articles_key(meta_recent_df, user_seen_magazines, 'magazine_id')

                popular_following_articles, _ = self._filter_df_articles(most_popular_df, user_following_authors)

                older_following_articles, _ = self._filter_df_articles(meta_older_df, user_following_authors)

                for article in recent_following_articles:
                    if article not in seens and article not in checkrec100:

                        checkrec100.add(article)
                        rec100.append(article)
                    if len(rec100) == self.topn:
                        break


                if len(checkrec100) < self.topn:
                    for article in recent_seen_authors_articles:
                        if article not in checkrec100 and article not in seens:
                            checkrec100.add(article)
                            rec100.append(article)
                        if len(rec100) == self.topn:
                            break                        

                if len(checkrec100) < self.topn:
                    for article in recent_seen_magazine_articles:
                        if article not in checkrec100 and article not in seens:
                            checkrec100.add(article)
                            rec100.append(article)
                        if len(rec100) == self.topn:
                            break

                if len(checkrec100) < self.topn:
                    for article in older_following_articles:
                        if article not in seens and article not in checkrec100:
                            checkrec100.add(article)
                            rec100.append(article)
                        if len(rec100) == self.topn:
                            break

############################################# fill with popular

                if len(checkrec100) < self.topn:
                    for popular_article in popular_following_articles:
                        if popular_article not in seens and popular_article not in checkrec100:

                            checkrec100.add(popular_article)
                            rec100.append(popular_article)
                        if len(rec100) == self.topn:
                            break

                if len(checkrec100) < self.topn:
                    for popular_article in most_popular:
                        if popular_article not in seens and popular_article not in checkrec100:
                            checkrec100.add(popular_article)
                            rec100.append(popular_article)
                        if len(rec100) == self.topn:
                            break
                fout.write(f'{user} {" ".join(rec100[:self.topn])}\n')

if __name__ == '__main__':
    # fire.Fire(RecommendMarch) # 명령어 직접 넣어서 돌리고 싶은경우 
    # RecommendMarch(from_dtm='2019021400', to_dtm='2019030100').recommend(os.path.join(conf.data_root, 'predict/dev.users'), 'recommend-3000.txt') # 3천명 파일 생성
    RecommendMarch(from_dtm='2019021400', to_dtm='2019030100').recommend(os.path.join(conf.data_root, 'predict/test.users'), 'recommend.txt') # 5천명 파일 생성