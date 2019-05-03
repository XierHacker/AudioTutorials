import numpy as np
import pandas as pd

en_chars=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","'"]
timit_phns = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx',
       'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix',
       'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
       'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

#print("timits_phn.size:",len(timit_phns))


def generate_index_files(key_list,save_path):
    '''
    制作关键字索引csv
    :param key_list:
    :param save_path:
    :return:
    '''
    ids=list(range(1,len(key_list)+1))
    print("ids:",ids)
    pd.DataFrame(data={"key":key_list,"id":ids}).to_csv(path_or_buf=save_path,index=False,encoding="utf-8")


def get_mapper(index_file):
    '''
    得到索引map
    :param index_file:
    :return:
    '''
    df=pd.read_csv(filepath_or_buffer=index_file,encoding="utf-8")
    key2id=pd.Series(data=df["id"].values,index=df["key"].values)
    id2key=pd.Series(data=df["key"].values,index=df["id"].values)
    # print("key2id:",key2id["aa"])
    #print("id2key:",id2key[1])
    return key2id,id2key

def padding(x,max_size):
    '''
    按照最大步长对x进行padding
    :param x: 待padding输入
    :param max_size: 最大尺寸
    :return: padding之后的结果
    '''
    pass







if __name__=="__main__":
    #generate_index_files(key_list=timit_phns,save_path="./timit_phns.csv")
    get_mapper(index_file="./timit_phns.csv")