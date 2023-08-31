import os
import time
import hashlib
import pickle
from geopy.geocoders import Photon

class CachedPhoton(Photon):
    def __init__(self, user_cache=True,cache_path='cache',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = user_cache
        self.cache_path = cache_path

    def geocode(self, location, language='en', location_bias=None):
        if self.use_cache:
            if not os.path.exists(self.cache_path):
                os.mkdir(self.cache_path)

            cache_key = self._get_cache_key(location, language, location_bias)
            cache_path = os.path.join(self.cache_path, f'{cache_key}.pickle')

            if os.path.exists(cache_path):
                #print('us_cache')
                with open(cache_path, 'rb') as cache_file:
                    return pickle.load(cache_file)

        time.sleep(0.5)  # APIのリクエストに似た処理を遅らせるためのダミー処理
        
        result = super().geocode(query=location, language=language, location_bias=location_bias)

        if self.use_cache and result:
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(result, cache_file)

        return result

    def _get_cache_key(self, location, language, location_bias):
        # キャッシュのキーとしてユニークな文字列を生成するロジック
        # ここでは単純に入力値を連結した文字列を返すが、実際の実装によって適切なキーを生成する必要がある
        return hashlib.sha256(f'{location}{language}{location_bias}'.encode()).hexdigest()
from geocode import CachedPhoton
import time
class LocationBiasManager(CachedPhoton):
    '''
    sample
    geoencoder = LocationBiasManager(target_country='United States', language='en', use_cache=False)
    testresult = geoencoder.geocode_with_bias('eastern montana', location_bias='montana')
    '''
    def __init__(self, target_country=None, language='en', use_cache=True, *args, **kwargs):
        super().__init__(user_cache=use_cache, *args, **kwargs)
        self.language = language
        self.country_bias = None
        if isinstance(target_country, str):
            self.country = target_country
            self.country_bias = super().geocode(location=target_country,language=self.language).point
        self.location_bias_dict = {}
    
    def _get_state_bias(self, state):        
        if state in self.location_bias_dict:
            return self.location_bias_dict[state]
        else:
            bias = super().geocode(location=state, language=self.language).point
            self.location_bias_dict[state] = bias
        return bias

    def make_querys(self,region,state=None,country=None):
        make_query = lambda x,y: ','.join([x,y]) if not y is None else x
        #優先順位の高い順にクエリを作成
        query_list = [make_query(region,None)]
        query_list.append(make_query(region,state))
        query_list.append(make_query(region,country))
        return query_list

    def __are_place_names_equal(self,name1, name2):
        return name1.strip().casefold() == name2.strip().casefold()

    def geocode_with_bias(self, region, location_bias=None):
            #指定したlocation_biasと出力が違う場合、location_biasに合うようにキューを書き換えるようにする
            state = ''
            if isinstance(location_bias, str):
                state = location_bias
                location_bias = self._get_state_bias(location_bias)

            for query in self.make_querys(region,state,self.country):
                #print(query)
                time.sleep(0.5)
                self.result = super().geocode(location=query, language=self.language, location_bias=location_bias)
                #検索結果がstateかcountryと一致するか確認
                #stateがある場合
                if self.result is None:
                    continue
                elif 'state' in self.result.raw['properties'].keys():
                    #print(self.result.raw['properties']['state'])
                    if self.__are_place_names_equal(self.result.raw['properties']['state'],state):
                        #print('ok same state region:{}  state:{} '.format(region,state))
                        break
                #stateがない場合
                elif 'country' in self.result.raw['properties'].keys() and state == '':
                    #print(self.result.raw['properties']['country'])
                    if self.__are_place_names_equal(self.result.raw['properties']['country'],self.country):
                        #print('ok same country region:{}  state:{} '.format(region,state))
                        break
                
            #print('ok')
            return self.result

import re
import numpy as np
import pandas as pd
import collections
import re
import os
def _make_query(region,state=None):
    split_marker = '/|-'
    make_query = lambda x,y: {'region':x,'location_bias':y}
    querys = [(region,None)]
    if not state is None:
        querys.append((region,state))
    query_list = []
    for region,state in reversed(querys):
        query_list.append([make_query(region,state)])
        if not re.search('/|-',region):
            continue
        split_region = re.split(split_marker,region)
        query_list_ = []
        for region_i in split_region:
            query_list_.append(make_query(region_i,state))
        query_list.append(query_list_)
    return query_list

def get_geocode_result(geoencoder,query):
    g = geoencoder.geocode_with_bias(**query)
    result_dict = dict()
    if not g is None:
        result_dict['lat'] = g.latitude
        result_dict['lng'] = g.longitude
        try:
            result_dict['country'] = g.raw['properties']['country']
        except:
            result_dict['country'] = None
        try:
            result_dict['state'] = g.raw['properties']['state']
        except:
            result_dict['state'] = None
        return result_dict
    return None

def get_geocoding(geoencoder,region, state=None):
    """
    Perform geocoding using OpenStreetMap (OSM) API.
    """
    
    #優先順位が高い順にクエリリストを作成
    query_list = _make_query(region,state)

    #クエリリストを優先順位順に検索,結果が得られたらループを抜ける
    for querys in query_list:
        rets = []
        for query in querys:
            #print(query)
            ret = get_geocode_result(geoencoder,query)
            if ret is not None:
                rets.append(ret)
            else:
                continue
        av_lat = [ret['lat']  if 'lat' in ret.keys() else np.nan for ret in rets]
        av_lng = [ret['lng'] if 'lng' in ret.keys() else np.nan for ret in rets]
        av_state = [ret['state'] if 'state' in ret.keys() else '' for ret in rets]
        av_country = [ret['country'] if 'country' in ret.keys() else '' for ret in rets]
        
        #検索結果が得られたらループを抜ける
        if len(rets) > 0:
            #検索結果の平均を取る
            av_lat = np.mean(av_lat)
            av_lng = np.mean(av_lng)
            # 最短の文字列を取得
            f = lambda s_list:max(dict(collections.Counter(s_list)).items(),key=lambda x:x[1])[0]
            av_state = f(av_state)
            av_country = f(av_country)
            return pd.Series({
                    'region': region,
                    'state': av_state,
                    'country': av_country,
                    'lat': av_lat,
                    'lng': av_lng
                })
    
    # If no valid results are found, return a Series with NaN values
    return pd.Series({
        'region': region,
        'state': np.nan,
        'country': np.nan,
        'lat': np.nan,
        'lng': np.nan
    })
#a = get_geocoding('saginaw-midland-baycity','michigan',use_cache=False)

import logging
from geocode import LocationBiasManager
from tqdm import tqdm
def geocoding(region_state_dict:dict,use_cache=True) -> pd.DataFrame:
    '''
        region_state_dict = {'region1':['state1','state2'],'region2':['state2'],'region3':[]'}
    '''
    # ログの出力名を設定
    logger = logging.getLogger('DistanceLogg')
    fh = logging.FileHandler('cal_distance_error.log')
    result = list()
    
    geoencoder = LocationBiasManager(target_country='United States',language='en',use_cache=use_cache)
    for region,states in tqdm(region_state_dict.items()):
        try:
            if len(states) == 0:
                result.append(get_geocoding(geoencoder,region,None))
                continue
            for state in states:
                result.append(get_geocoding(geoencoder,region,state))
        except:
            logger.exception('Error in geocoding region,state :',region,state)
    return pd.DataFrame(result)

import folium
def plot_folium(locations_df):
    #サイズを指定する
    folium_figure = folium.Figure(width=1500, height=700)

    # 初期表示の中心の座標を指定して地図を作成する。
    center_lon=locations_df["lot"].mean()
    center_lat=locations_df["lat"].mean()
    folium_map = folium.Map([center_lat,center_lon],
    zoom_start=4.5).add_to(folium_figure)
    for i in range(len(locations_df)):
        print('\r {}'.format(i),end='')
        try:
            lot =locations_df.loc[i, "lot"]
            lat = locations_df.loc[i, "lat"]
            region = locations_df.loc[i, "region"]
            folium.Marker(location=[lat,lot],popup=region).add_to(folium_map)
        except:
            continue
    return folium_map