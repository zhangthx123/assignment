# -*- coding: utf-8 -*-
import pandas as pd
import data_class
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
import copy
from heapq import heappush, heappop


start = time.process_time()
    
# 创建实例
truck = data_class.Truck()
order = data_class.Order()

# 输入
truck.V_truck = pd.read_excel(r'truck_5.xlsx')['V_truck'].values
truck.W_truck = pd.read_excel(r'truck_5.xlsx')['W_truck'].values
truck.Cost_truck = pd.read_excel(r'truck_5.xlsx')['Cost_truck'].values
truck.Index_truck = pd.read_excel(r'truck_5.xlsx')['Index'].values
truck.Rv_truck = pd.read_excel(r'truck_5.xlsx')['V_truck'].values
truck.Rw_truck = pd.read_excel(r'truck_5.xlsx')['W_truck'].values

order.V_order = pd.read_excel(r'order_5.xlsx')['V_order'].values  
order.W_order = pd.read_excel(r'order_5.xlsx')['W_order'].values  
order.Time_earliest = pd.read_excel(r'order_5.xlsx')['Time_earliest'].values 
order.Time_latest = pd.read_excel(r'order_5.xlsx')['Time_latest'].values 
order.Index_order = pd.read_excel(r'order_5.xlsx')['Index'].values

df_coordinates = pd.read_excel(r'distance.xlsx', header=None)
array_coordinates = df_coordinates.values  # dataframe变为array, 是一个向量矩阵,代表4S店坐标
array_distance = pdist(array_coordinates,metric='euclidean')  # euclidean代表欧式距离
square_distance = squareform(array_distance)  # 将distA数组变成一个矩阵
df_distance = pd.DataFrame(square_distance)  # array变为dataframe

# 顺序
order.Index_order = order.Index_order.tolist()


# 给定参数         
r_dist_unit_cost = 10  # 单位距离的成本为10  r_dist_unit_cost
v_truck_speed = 60  # 车辆行驶速度为60 v_truck_speed
lambda1_visiting_total_dist = 1  # lambda1_visiting_total_dist
lambda2_truck_total_cost = 1  # lambda2_truck_total_cost


class BnBNode:
    def __init__(self):
        self.Parent = None
        self.level = 0
        self.order = order.Index_order[0]
        self.weight = 0
        self.volume = 0
        self.truck = truck.Index_truck[0]
        self.cost = 0
        self.Rv_truck = copy.deepcopy(truck.Rv_truck)  # 各车剩余的体积
        self.Rw_truck = copy.deepcopy(truck.Rw_truck)
        self.truck_last_order = [0 for i in range(len(truck.Index_truck))]  # 各车的上一个订单
        self.T_truck = [0 for i in range(len(truck.Index_truck))]   # 每辆车上一个订单的结束时间
        self.expanded = False
        
    def Expand(self):  # 生成能向下一层分支的节点list
        node_list = []
        truck_list = []
        l = 0  
        while l < len(truck.Index_truck):
#            i = -np.argsort(self.Rv_truck)[l]  # 从大到小排序
            i = np.argsort(self.Rv_truck)[l]  # 从小到大排序
            if self.Rv_truck[i] >= order.V_order[order.Index_order[self.level + 1]] and self.Rw_truck[i] >= order.W_order[order.Index_order[self.level + 1]] \
            and order.Time_earliest[order.Index_order[self.level + 1]] <= self.T_truck[i] + df_distance.iloc[self.truck_last_order[i],order.Index_order[self.level + 1]]/v_truck_speed \
            and self.T_truck[i] + df_distance.iloc[self.truck_last_order[i],order.Index_order[self.level + 1]]/v_truck_speed <= order.Time_latest[order.Index_order[self.level + 1]]:
                node_take = BnBNode()  # 生成子节点
                node_take.Parent = self
                node_take.level  = self.level + 1  
                node_take.order = order.Index_order[node_take.level]  # level0是第0号订单
                node_take.weight = order.W_order[node_take.order]
                node_take.volume = order.V_order[node_take.order]
                node_take.truck = truck.Index_truck[i]
                
                node_take.Rv_truck = copy.deepcopy(self.Rv_truck)  #自调用
                node_take.Rw_truck = copy.deepcopy(self.Rw_truck)
                node_take.Rv_truck[i] = self.Rv_truck[i] - node_take.volume  # 更新剩余体积
                node_take.Rw_truck[i] = self.Rw_truck[i] - node_take.weight  # 更新剩余重量

                if self.truck_last_order[i] == 0: 
                    node_take.cost = copy.deepcopy(self.cost + lambda1_visiting_total_dist*r_dist_unit_cost*df_distance.iloc[self.truck_last_order[i],node_take.order] + lambda2_truck_total_cost*truck.Cost_truck[i])
                else: 
                    node_take.cost = copy.deepcopy(self.cost + lambda1_visiting_total_dist*r_dist_unit_cost*df_distance.iloc[self.truck_last_order[i],node_take.order])
                node_take.truck_last_order[i] = node_take.order   # 更新上一个订单
                node_take.T_truck[i] = self.T_truck[i] + df_distance.iloc[self.truck_last_order[i],node_take.order]/v_truck_speed
                
                self.expanded = True
                node_list.append(copy.deepcopy(node_take))   
                truck_list.append(truck.Index_truck[i])  # 分支产生的新节点所用的truck
            l = l + 1    
          
        return node_list, truck_list


def solve():
    root = BnBNode()  # 生成根节点
    nodes = []  # 处理过的总节点
    nodes.append(root)
    i = 0   
    heap = []  # 待处理节点
    heappush(heap, (-1.0*root.level, 0))  # 入堆.[(1, 0), (2, 0), (2, 1), (2, 2)] 默认从小到大所以要乘-1
    idx_to_expand = 0  #初始值
    UB = 41005.6
    optimal_node = BnBNode()
#    print('--------------start')
    while heap:
        idx_to_expand = heappop(heap)[1]  # 出堆 
        node_to_expand = nodes[idx_to_expand]
        df_tmp = df_distance.replace(0,np.nan)
        df_use = df_tmp.min(axis=1)
        c_min = 0
        for i in range(node_to_expand.level, len(order.Index_order)):
            c_min = c_min + df_use.iloc[i]  # 第i + 1行到最后    
        if node_to_expand.cost + c_min >= UB:  # 剪支
            continue
        elif node_to_expand.level + 1 == len(order.Index_order):  # 到达最后一层计算成本，判断是否要更新ub          
            if UB > node_to_expand.cost:
                UB = node_to_expand.cost
                optimal_node = copy.deepcopy(node_to_expand)
        else:  # 分支
            node_list, truck_list = node_to_expand.Expand()  # 分支后新加入的节点
            for i in range(len(node_list)):
                nodes.append(node_list[i])
                heappush(heap, (-1.0 * node_list[i].level, len(nodes) - 1))  # 将分支产生的新节点入堆（因为nodes中包含根节点，所以要-1）
 
    cost = optimal_node.cost
    sol = np.zeros((len(order.Index_order)-1,len(truck.Index_truck)))
    tmp = copy.deepcopy(optimal_node)
    while optimal_node.Parent:
        sol[optimal_node.order-1][optimal_node.truck] = 1
        optimal_node = optimal_node.Parent
    return sol, cost, nodes,tmp

sol,cost,nodes,optimal_node = solve()
sol = pd.DataFrame(sol)    
sol.to_excel('sol.xlsx')      
      
end = time.process_time()
print("运行时间为%.03f秒" %(end-start)) 
                       
     
             