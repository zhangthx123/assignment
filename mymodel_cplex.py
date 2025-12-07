import gurobipy as gp
from gurobipy import GRB, quicksum

import pandas as pd
import data_class
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

start = time.process_time()  # 将代码改为Gurobi版本

# ======================
# 创建实例 & 读入数据
# ======================
truck = data_class.Truck()
order = data_class.Order()

truck.V_truck = pd.read_excel(r'data\truck_5.xlsx')['V_truck'].tolist()
truck.W_truck = pd.read_excel(r'data\truck_5.xlsx')['W_truck'].tolist()
truck.Cost_truck = pd.read_excel(r'data\truck_5.xlsx')['Cost_truck'].tolist()

order.V_order = pd.read_excel(r'data\order_5_test.xlsx')['V_order'].tolist()
order.W_order = pd.read_excel(r'data\order_5_test.xlsx')['W_order'].tolist()
order.Time_earliest = pd.read_excel(r'data\order_5_test.xlsx')['Time_earliest'].tolist()
order.Time_latest = pd.read_excel(r'data\order_5_test.xlsx')['Time_latest'].tolist()

# 数据初始化处理
n = len(order.V_order) - 1  # 订单 i,k (0 是仓库)
m = len(truck.V_truck)      # 辆车 j

r = 10   # 单位距离成本
v = 60   # 速度
lambda1 = 1
lambda2 = 1
M = 10000

Location_Order_index_I = range(0, n + 1)  # 0 是仓库，1..n 为订单
Truck_index_J = range(0, m)

# ======================
# 距离矩阵
# ======================
df_coordinates = pd.read_excel(r'data\distance_20.xlsx', header=None)
array_coordinates = df_coordinates.values
array_distance = pdist(array_coordinates, metric='euclidean')
square_distance = squareform(array_distance)
df_distance = pd.DataFrame(square_distance)
Distance_locationOrder = {
    (i, k): df_distance.iloc[i, k]
    for i in Location_Order_index_I for k in Location_Order_index_I
}

# ======================
# 访问顺序矩阵
# ======================
order.Index_order = pd.read_excel(r'data\order_5_test.xlsx')['Index'].values.tolist()

df_sigma_list = order.Index_order
df_sigma = np.zeros((len(df_sigma_list), len(df_sigma_list)))
for i in range(len(df_sigma_list) - 1):
    for j in range(i + 1, len(df_sigma_list)):
        df_sigma[df_sigma_list[i], df_sigma_list[j]] = 1
df_sigma = pd.DataFrame(df_sigma)

VistingSequence_matrix_location = {
    (i, k): df_sigma.iloc[i, k]
    for i in Location_Order_index_I for k in Location_Order_index_I
}

# ======================
# 创建 Gurobi 模型
# ======================
model = gp.Model("IP_Model_Gurobi")

# 决策变量
x = model.addVars(
    Location_Order_index_I, Truck_index_J,
    vtype=GRB.BINARY,
    name="x"
)

# 中间变量：三维
y = model.addVars(
    Location_Order_index_I, Location_Order_index_I, Truck_index_J,
    vtype=GRB.BINARY,
    name="y"
)

z = model.addVars(
    Location_Order_index_I, Location_Order_index_I, Truck_index_J,
    vtype=GRB.BINARY,
    name="z"
)

s = model.addVars(
    Location_Order_index_I, Location_Order_index_I, Truck_index_J,
    vtype=GRB.CONTINUOUS,
    name="s"
)

f = model.addVars(
    Location_Order_index_I, Location_Order_index_I, Truck_index_J,
    vtype=GRB.BINARY,
    name="f"
)

u = model.addVars(
    Truck_index_J,
    vtype=GRB.BINARY,
    name="u"
)

t = model.addVars(
    Location_Order_index_I,
    vtype=GRB.CONTINUOUS,
    name="t"
)

# ======================
# 约束
# ======================

# 1) x 与 y 的关系
model.addConstrs(
    (2 * y[i, k, j] <= x[i, j] + x[k, j]
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="link_y_x1"
)

model.addConstrs(
    (x[i, j] + x[k, j] <= y[i, k, j] + 1
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="link_y_x2"
)

# 2) y, z 与访问顺序、f 的关系
model.addConstrs(
    (2 * z[i, k, j] <= y[i, k, j] + VistingSequence_matrix_location[i, k]
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="link_z_y1"
)

model.addConstrs(
    (y[i, k, j] + VistingSequence_matrix_location[i, k]
     <= z[i, k, j] + 1 + f[i, k, j]
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="link_z_y2"
)

# 3) 老师新加的那组约束
model.addConstrs(
    (quicksum(
        x[h, j] * VistingSequence_matrix_location[i, h] * VistingSequence_matrix_location[h, k]
        for h in range(1, n + 1) if h != i and h != k
    ) <= M * f[i, k, j] + M * (1 - y[i, k, j])
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="new_cons1"
)

model.addConstrs(
    (f[i, k, j] - M * (1 - y[i, k, j])
     <= quicksum(
        x[h, j] * VistingSequence_matrix_location[i, h] * VistingSequence_matrix_location[h, k]
        for h in range(1, n + 1) if h != i and h != k
     )
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="new_cons2"
)

model.addConstrs(
    (f[i, k, j] <= y[i, k, j]
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="new_cons3"
)

model.addConstrs(
    (z[i, k, j] <= 1 - f[i, k, j]
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="new_cons4"
)

# 4) 容积约束
model.addConstrs(
    (quicksum(x[i, j] * order.V_order[i] for i in Location_Order_index_I)
     <= u[j] * truck.V_truck[j]
     for j in Truck_index_J),
    name="capacity_volume"
)

# 5) 载重约束
model.addConstrs(
    (quicksum(x[i, j] * order.W_order[i] for i in Location_Order_index_I)
     <= u[j] * truck.W_truck[j]
     for j in Truck_index_J),
    name="capacity_weight"
)

# 6) 时间窗约束
# earliest <= 到达时间
model.addConstrs(
    (order.Time_earliest[k] <= quicksum(
        s[i, k, j] for i in Location_Order_index_I for j in Truck_index_J if i != k
    )
     for k in Location_Order_index_I),
    name="time_window_earliest"
)

# 到达时间 <= latest
model.addConstrs(
    (quicksum(
        s[i, k, j] for i in Location_Order_index_I for j in Truck_index_J if i != k
    ) <= order.Time_latest[k]
     for k in Location_Order_index_I),
    name="time_window_latest"
)

# 定义 t[k]
model.addConstrs(
    (t[k] == quicksum(
        s[i, k, j] for i in Location_Order_index_I for j in Truck_index_J if i != k
    )
     for k in Location_Order_index_I),
    name="def_t"
)

# s 与 z 的关系
model.addConstrs(
    (s[i, k, j] <= M * z[i, k, j]
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="s_leq_Mz"
)

model.addConstrs(
    (s[i, k, j] <= t[i] + (Distance_locationOrder[i, k] / v)
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="s_leq_ti_plus_travel"
)

model.addConstrs(
    (s[i, k, j] >= t[i] + (Distance_locationOrder[i, k] / v) - M * (1 - z[i, k, j])
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="s_geq_ti_plus_travel_bigM"
)

model.addConstrs(
    (s[i, k, j] >= 0
     for i in Location_Order_index_I
     for k in Location_Order_index_I
     for j in Truck_index_J if i != k),
    name="s_nonneg"
)

# 7) 每个订单只分配给一辆车（不含仓库0）
model.addConstrs(
    (quicksum(x[i, j] for j in Truck_index_J) == 1
     for i in range(1, n + 1)),
    name="assign_each_order"
)

# 8) 使用的车辆必须装载订单0
model.addConstrs(
    (u[j] == x[0, j] for j in Truck_index_J),
    name="truck_use_if_depot"
)

# ======================
# 目标函数
# ======================
objective_1 = quicksum(
    z[i, k, j] * Distance_locationOrder[i, k] * r
    for i in Location_Order_index_I
    for k in Location_Order_index_I
    for j in Truck_index_J
)

objective_2 = quicksum(
    u[j] * truck.Cost_truck[j] for j in Truck_index_J
)

model.setObjective(lambda1 * objective_1 + lambda2 * objective_2, GRB.MINIMIZE)

# ======================
# 求解
# ======================
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
else:
    print("Model status:", model.status)

end = time.process_time()
print("运行时间为%.03f秒" % (end - start))

# ======================
# 将二维变量结果导出为矩阵 & Excel
# ======================
def matrix_solution_to_dataframe(var_matrix):
    """
    var_matrix: Gurobi 的二维变量字典，例如 x[i,j]
    返回：pandas DataFrame
    """
    keys = list(var_matrix.keys())
    row_indices = sorted({i for (i, j) in keys})
    col_indices = sorted({j for (i, j) in keys})

    data_dict = {}
    for j in col_indices:
        col_dict = {}
        for i in row_indices:
            col_dict[i] = var_matrix[i, j].X  # 解的值
        data_dict[j] = col_dict

    try:
        from pandas import DataFrame
        return DataFrame(data_dict)
    except ImportError:
        print("-- pandas not found, returning a dict")
        return data_dict

if model.status == GRB.OPTIMAL:
    df = matrix_solution_to_dataframe(x)
    df.to_excel('x_vars_gurobi.xlsx')
