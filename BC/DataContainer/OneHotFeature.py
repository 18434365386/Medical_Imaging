'''.
Feb 2, 2019.
Yang SONG, songyangmri@gmail.com
'''

import pandas as pd
from copy import deepcopy
from BC.DataContainer.DataContainer import DataContainer


class FeatureEncodingOneHot():
    def __init__(self):
        pass

    def OneHotOneColumn(self, data_container, feature_list):
        info = data_container.GetFrame()
        feature_name = data_container.GetFeatureName()
        for feature in feature_list:
            assert(feature in feature_name)

        new_info = pd.get_dummies(info, columns=feature_list)
        new_data = DataContainer()
        new_data.SetFrame(new_info)
        return new_data



if __name__ == '__main__':
    # 1、深拷贝数据
    data = DataContainer()
    data.Load(r'c:\Users\yangs\Desktop\test.csv')
    info = data.GetFrame()

    # 2、深拷贝后的数据进行哑变量转换并输出到固定路径
    new_info = pd.get_dummies(info, columns=['bGs', 'PIRADS', 't2score', 'DWIscore', 'MR_stage']) #要编码的 DataFrame 中的列名
    new_info.to_csv(r'c:\Users\yangs\Desktop\test_onehot.csv')