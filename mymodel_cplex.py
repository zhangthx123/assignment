# -*- coding: utf-8 -*-
import docplex.mp.model as cpx
import pandas as pd
import data_class
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

start = time.process_time()

# 创建实例
truck = data_class.Truck()
order = data_class.Order()

# 赋值
truck.V_truck = pd.read_excel(r'truck_5.xlsx')['V_truck'].tolist()
truck.W_truck = pd.read_excel(r'truck_5.xlsx')['W_truck'].tolist()
truck.Cost_truck = pd.read_excel(r'truck_5.xlsx')['Cost_truck'].tolist()

order.V_order = pd.read_excel(r'order_5.xlsx')['V_order'].tolist()  
order.W_order = pd.read_excel(r'order_5.xlsx')['W_order'].tolist()  
order.Time_earliest = pd.read_excel(r'order_5.xlsx')['Time_earliest'].tolist() 
order.Time_latest = pd.read_excel(r'order_5.xlsx')['Time_latest'].tolist() 

#数据初始化处理
n = len(order.V_order) - 1  # 订单i,k
m = len(truck.V_truck)  # 辆车j

r = 10  # 单位距离的成本为10
v = 60  # 车辆行驶速度为60
lambda1 = 1
lambda2 = 1
M = 10000

Location_Order_index_I = range(0, n+1)  #python range左闭右开 0是仓库，n个订单
Truck_index_J = range(0, m)

# 输入距离坐标，输出距离矩阵
df_coordinates = pd.read_excel(r'distance.xlsx', header=None)
array_coordinates = df_coordinates.values  # dataframe变为array, 是一个向量矩阵,代表4S店坐标
array_distance = pdist(array_coordinates,metric='euclidean')  # euclidean代表欧式距离
square_distance = squareform(array_distance)  # 将distA数组变成一个矩阵
df_distance = pd.DataFrame(square_distance)  # array变为dataframe
Distance_locationOrder = {(i,k):df_distance.iloc[i,k] for i in Location_Order_index_I for k in Location_Order_index_I}  # 4S店S_i与S_k间的距离
# 顺序
order.Index_order = pd.read_excel(r'order_5.xlsx')['Index'].values
order.Index_order = order.Index_order.tolist()
#order.Index_order.remove(0)  
#order.Index_order.insert(0,0)  # 在序列最开始加0，即均从仓库出发


# 输入顺序序列，输出顺序0-1矩阵
df_sigma_list = order.Index_order
df_sigma = np.zeros((len(df_sigma_list),len(df_sigma_list)))
for i in range(len(df_sigma_list)-1):
    for j in range(i+1,len(df_sigma_list)):
        df_sigma[df_sigma_list[i],df_sigma_list[j]] = 1
df_sigma = pd.DataFrame(df_sigma)
VistingSequence_matrix_location = {(i,k):df_sigma.iloc[i,k] for i in Location_Order_index_I for k in Location_Order_index_I}  # 固定的访问顺序


# 创建模型
opt_model = cpx.Model(name="IP Model")
# 决策变量
x_vars  = {(i,j): opt_model.binary_var(name="x_{}_{}".format(i,j)) for i in Location_Order_index_I for j in Truck_index_J}              

# 中间变量
y_vars = opt_model.binary_var_cube(Location_Order_index_I,Location_Order_index_I,Truck_index_J,name='y')
z_vars = opt_model.binary_var_cube(Location_Order_index_I,Location_Order_index_I,Truck_index_J,name='z')
s_vars = opt_model.continuous_var_cube(Location_Order_index_I,Location_Order_index_I,Truck_index_J,name='s')
f_vars = opt_model.binary_var_cube(Location_Order_index_I,Location_Order_index_I,Truck_index_J,name='f')
u_vars = {j: opt_model.binary_var(name="u_{}".format(j)) for j in Truck_index_J}
t_vars = {i: opt_model.continuous_var(name="t_{}".format(i)) for i in Location_Order_index_I}

# 约束
opt_model.add_constraints(2*y_vars[i,k,j] <= (x_vars[i,j]+x_vars[k,j]) for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
opt_model.add_constraints((x_vars[i,j]+x_vars[k,j]) <= y_vars[i,k,j]+1 for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
# 修改
opt_model.add_constraints(2*z_vars[i,k,j] <= (y_vars[i,k,j]+VistingSequence_matrix_location[i,k]) for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
opt_model.add_constraints((y_vars[i,k,j]+VistingSequence_matrix_location[i,k]) <= z_vars[i,k,j] + 1 + f_vars[i,k,j] for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)

# 老师新加
opt_model.add_constraints((opt_model.sum(x_vars[h,j] * VistingSequence_matrix_location[i,h] * VistingSequence_matrix_location[h,k] for h in range(1, n+1) if h != i and h != k)) <= (M*f_vars[i,k,j] + M*(1-y_vars[i,k,j])) for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
opt_model.add_constraints((f_vars[i,k,j] - M*(1-y_vars[i,k,j])) <= (opt_model.sum(x_vars[h,j] * VistingSequence_matrix_location[i,h] * VistingSequence_matrix_location[h,k] for h in range(1, n+1) if h != i and h != k)) for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
opt_model.add_constraints(f_vars[i,k,j] <= y_vars[i,k,j] for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
opt_model.add_constraints(z_vars[i,k,j] <= (1 - f_vars[i,k,j]) for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)

# 容积约束
opt_model.add_constraints(opt_model.sum(x_vars[i,j]*order.V_order[i] for i in Location_Order_index_I) <= u_vars[j]*truck.V_truck[j] for j in Truck_index_J)
# 载重量约束
opt_model.add_constraints(opt_model.sum(x_vars[i,j]*order.W_order[i] for i in Location_Order_index_I) <= u_vars[j]*truck.W_truck[j] for j in Truck_index_J)
# 时间窗约束
opt_model.add_constraints(order.Time_earliest[k] <= opt_model.sum(s_vars[i,k,j] for i in Location_Order_index_I for j in Truck_index_J if i != k) for k in Location_Order_index_I)
opt_model.add_constraints(opt_model.sum(s_vars[i,k,j] for i in Location_Order_index_I for j in Truck_index_J if i != k) <= order.Time_latest[k] for k in Location_Order_index_I)
opt_model.add_constraints(t_vars[k] == opt_model.sum(s_vars[i,k,j] for i in Location_Order_index_I for j in Truck_index_J if i != k) for k in Location_Order_index_I)
opt_model.add_constraints(s_vars[i,k,j] <= M*z_vars[i,k,j] for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
opt_model.add_constraints(s_vars[i,k,j] <= t_vars[i] + (Distance_locationOrder[i,k]/v) for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
opt_model.add_constraints(s_vars[i,k,j] >= t_vars[i] + (Distance_locationOrder[i,k]/v) - M*(1-z_vars[i,k,j]) for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)
opt_model.add_constraints(s_vars[i,k,j] >= 0 for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J if i != k)

# 每个4S店订单只指派给一辆车
opt_model.add_constraints(opt_model.sum(x_vars[i,j] for j in Truck_index_J)==1 for i in range(1,n+1))
# 每辆用到的车必须装载订单0
opt_model.add_constraints(u_vars[j] == x_vars[0,j] for j in Truck_index_J)


# 目标函数分两部分
objective_1 = opt_model.sum(z_vars[i,k,j] * Distance_locationOrder[i,k] * r for i in Location_Order_index_I for k in Location_Order_index_I for j in Truck_index_J)
objective_2 = opt_model.sum(u_vars[j] * truck.Cost_truck[j] for j in Truck_index_J)

# for minimization
opt_model.minimize(lambda1*objective_1 + lambda2*objective_2)

# 模型求解
sol = opt_model.solve()

# 结果输出
print(sol) # 获取默认形式的输出
#print(sol.get_all_values()) # 获取所有的变量解
cost = sol.get_objective_value()  # 目标函数值
#print(x_vars[0,1].solution_value)  # 获取某一变量取值
#print(s_vars[0,0,0].solution_value)
#print(z_vars[1,3,3].solution_value)
#print(y_vars[1,3,3].solution_value)

end = time.process_time()
print("运行时间为%.03f秒" %(end-start)) 
 
## 二维变量可输出到矩阵
def matrix_solution_to_dataframe(var_matrix, sol):
    # compute a 2d dataframe from a variable matrix and a solution
    # (i, j) -> life][i,j].solution_value in one step.
    matrix_val_d = sol.get_value_dict(var_matrix)
    # extract rows and column indices
    keys = var_matrix.keys()
    row_indices = set()
    col_indices = set()
    for (i, j) in keys:
        row_indices.add(i)
        col_indices.add(j)
    # build a column-oriented dict:
    dtf_d = {col: {row: matrix_val_d[row, col] for row in row_indices} for col in col_indices}
    try:
        from pandas import DataFrame
        return DataFrame(dtf_d)
    except ImportError:
        print(f"-- pandas not found, returning a dict")
        return dtf_d
#    
df = matrix_solution_to_dataframe(x_vars,sol)    
#    
df.to_excel('x_vars.xlsx')

