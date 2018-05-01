import numpy
import random
import pandas as pd
import scipy.special as special
import math
from math import log

from collections import Counter
import pandas as pd

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        #产生样例数据
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)


    def __fixed_point_iteration(self, tries, success, alpha, beta):
        #迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries-success+beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries+alpha+beta) - special.digamma(alpha+beta)).sum()

        # for i in range(len(tries)):
        #     sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
        #     sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
        #     sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update(self, tries, success, iter_num, epsilon):
        #更新策略
        '''
            tries ： 展示次数
           success : 点击次数
           iter_num : 迭代次数
           epsilon : 精度
        '''

        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            print(new_alpha, new_beta, i)
            self.alpha = new_alpha
            self.beta = new_beta


    # 平滑方式2
    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        # 求均值和方差
        mean, var = self.__compute_moment(tries, success)
        # print 'mean and variance: ', mean, var
        # self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        # self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)

    def __compute_moment(self, tries, success):
        # 求均值和方差
        '''moment estimation'''
        ctr_list = []
        # var = 0.0
        mean = (success / tries).mean()
        if len(tries) == 1:
            var = 0
        else:
            var = (success / tries).var()
        # for i in range(len(tries)):
        #     ctr_list.append(float(success[i])/tries[i])
        # mean = sum(ctr_list)/len(ctr_list)
        # for ctr in ctr_list:
        #     var += pow(ctr-mean, 2)
        return mean, var
