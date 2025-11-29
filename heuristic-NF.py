# -*- coding: utf-8 -*-
import pandas as pd
import data_class
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time

start = time.process_time()

# 创建实例
truck = data_class.Truck()
order = data_class.Order()

# 赋值

truck.V_truck = pd.read_excel(r'truck_5.xlsx')['V_truck'].values
truck.W_truck = pd.read_excel(r'truck_5.xlsx')['W_truck'].values
truck.Cost_truck = pd.read_excel(r'truck_5.xlsx')['Cost_truck'].values
truck.Index_truck = pd.read_excel(r'truck_5.xlsx')['Index'].values
truck.Order_truck = [[] for i in range(len(truck.Index_truck))]  # 初始订单集合为空集
truck.Rv_truck = pd.read_excel(r'truck_5.xlsx')['V_truck'].values
truck.Rw_truck = pd.read_excel(r'truck_5.xlsx')['W_truck'].values
truck.T_truck = [0 for i in range(len(truck.Index_truck))] 
truck.Order_last_truck = [0 for i in range(len(truck.Index_truck))] 

order.V_order = pd.read_excel(r'order_5.xlsx')['V_order'].values  
order.W_order = pd.read_excel(r'order_5.xlsx')['W_order'].values  
order.Time_earliest = pd.read_excel(r'order_5.xlsx')['Time_earliest'].values 
order.Time_latest = pd.read_excel(r'order_5.xlsx')['Time_latest'].values 
order.Index_order = pd.read_excel(r'order_5.xlsx')['Index'].values

# 车辆初始化排序
#truck.Index_truck = np.argsort(-truck.V_truck)  # 将index的array变为按体积非升排序的array[3,5,2,4,1]
truck.Index_truck = np.argsort(truck.V_truck)  # 将index的array变为按体积非减排序的array

# 距离
df_coordinates = pd.read_excel(r'distance.xlsx', header=None)
#经度，纬度
array_coordinates = df_coordinates.values  # dataframe变为array, 是一个向量矩阵,代表4S店坐标
array_distance = pdist(array_coordinates,metric='euclidean')  # euclidean代表欧式距离
square_distance = squareform(array_distance)  # 将distA数组变成一个矩阵
df_distance = pd.DataFrame(square_distance)  # array变为dataframe
# 顺序
order.Index_order = order.Index_order.tolist()

r = 10  # 单位距离的成本为10
v = 20  # 车辆行驶速度为60
lambda1 = 1
lambda2 = 1

def NF_heuristic():  
    c = 0
    k = 1
    l = 0
    while k < len(order.Index_order):
        Flag = False
        while l < len(truck.Index_truck):
            if truck.Rv_truck[truck.Index_truck[l]] >= order.V_order[order.Index_order[k]] and truck.Rw_truck[truck.Index_truck[l]] >= order.W_order[order.Index_order[k]] \
            and order.Time_earliest[order.Index_order[k]] < truck.T_truck[truck.Index_truck[l]] + df_distance.iloc[truck.Order_last_truck[truck.Index_truck[l]],k]/v \
            and truck.T_truck[truck.Index_truck[l]] + df_distance.iloc[truck.Order_last_truck[truck.Index_truck[l]],k]/v < order.Time_latest[order.Index_order[k]]:
                if len(truck.Order_truck[truck.Index_truck[l]]) is 0: 
                    c = c + lambda1*r*df_distance.iloc[truck.Order_last_truck[truck.Index_truck[l]],order.Index_order[k]] + lambda2*truck.Cost_truck[truck.Index_truck[l]]
                else:
                    c = c + lambda1*r*df_distance.iloc[truck.Order_last_truck[truck.Index_truck[l]],order.Index_order[k]]
               
                truck.Order_truck[truck.Index_truck[l]].append(order.Index_order[k]) # 并集            
                truck.Rv_truck[truck.Index_truck[l]] = truck.Rv_truck[truck.Index_truck[l]] - order.V_order[order.Index_order[k]]
                truck.Rw_truck[truck.Index_truck[l]] = truck.Rw_truck[truck.Index_truck[l]] - order.W_order[order.Index_order[k]]
                truck.T_truck[truck.Index_truck[l]] = truck.T_truck[truck.Index_truck[l]] + df_distance.iloc[truck.Order_last_truck[truck.Index_truck[l]],order.Index_order[k]]/v
                truck.Order_last_truck[truck.Index_truck[l]] = order.Index_order[k]
                
                Flag = True
                break
            else:
                l = l + 1
        
        if Flag == False:print("order:"+str(k) + "无解")
        k += 1
    return c, truck.Order_truck

c, truck.Order_truck = NF_heuristic()
                    
end = time.process_time()
print("运行时间为%.03f秒" %(end-start))  # 程序开始到结束的运行时间
                       
                    
                    
                    
                    
                    
                    
                


    




    