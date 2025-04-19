from pyscipopt import Model, quicksum, Conshdlr, SCIP_RESULT
import pandas as pd
import networkx
import numpy as np

class TSPconshdlr(Conshdlr):
    def __init__(self, variables_x,variables_y,pos):
        self.variables_x = variables_x
        self.variables_y = variables_y
        self.pos = pos

    def find_subtours(self, solution=None):
        edges = []
        x = self.variables_x
        for (i,j) in x:
            if self.model.getSolVal(solution , x[i,j]) > 0.5:
                edges.append((i,j))
        G = networkx.Graph()
        G.add_edges_from(edges)
        components = list(networkx.connected_components(G))
        return [] if len(components) == 1 else components
    
    def conscheck(self, constraints, solution, checkintegrality, checklprows , printreason, completely):
        if self.find_subtours(solution):
            return {"result": SCIP_RESULT.INFEASIBLE}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}
        
    def consenfolp(self, constraints, n_useful_conss , sol_infeasible):
        subtours = self.find_subtours()
        if subtours:
            x = self.variables_x
            y = self.variables_y
            pos = self.pos
            for subset in subtours:
                if pos not in subset:    
                    self.model.addCons(quicksum(x[i,j] for i in subset for j in subset if i != j) <= quicksum(y[k] for k in subset) - 1)
                    print("cut: len(%s) <= %s" % (subset, len(subset) - 1))
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}
    
    def conslock(self, constraint, locktype, nlockspos , nlocksneg):
        x = self.variables_x
        y = self.variables_y
        for (i,j) in x:
            self.model.addVarLocks(x[i,j], nlocksneg, nlockspos)
        for k in y:
            self.model.addVarLocks(y[k], nlocksneg, nlockspos)


def my_GI_rule_scip(N, g, w, pre_idx, lat_idx, num = 8, width_gap=250):
    # 相较之前的版本，这里的N包含前一个外板和后一个外板，并且包含在这两个外板之间的所有特斯拉板
    # pre_idx是前一个外板的编号，lat_idx是后一个外板的编号
    # 根据我们的命题，我们知道外板必须是从宽到窄进行排列，因此我们从最大外板开始

    penalty = gap_score(w,width_gap)

    # 这里我们需要强制把pre_idx变成0，因为我们能在子回路约束回调时需要保留含有前一块外板的环，首先0编号不能在N中
    if 0 in N:
        print('需要预处理，因为第0个样材需要被用作虚拟材料')
        return None, None, None

    '''
    # 修改N
    N[N.index(pre_idx)] = 0

    # 修改索引
    g.rename(index={pre_idx: 0}, inplace=True)
    w.rename(index={pre_idx: 0}, inplace=True)
    '''

    model = Model('TSP')

    # 创建变量，这里我们不再需要辅助变量c
    x,y = {},{}

    # 路径变量x及选取变量y
    for i in N:
        y[i] = model.addVar(vtype="B", name="y(%s)" % (i))
        for j in N:
            if i != j:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)" % (i,j))

    # 必须保证前一个外板和后一个外板在排列中，并且我们虚拟后一个外板接到前一个外板，这里的惩罚为0
    # model.addCons(y[0] == 1)
    # model.addCons(y[lat_idx] == 1)
    model.addCons(x[lat_idx,pre_idx] == 1)

    # 流量平衡约束
    for i in N:
        model.addCons(quicksum(x[i,j] for j in N if j != i) == y[i])
        model.addCons(quicksum(x[j,i] for j in N if j != i) == y[i])

    # 特斯拉板数量约束
    model.addCons(quicksum(y[i] for i in N) == num+2) # 加2是因为我们包含了两个外板

    # 质量约束不再必要，因为我们是一种动态添加的方式

    # 回调子回路约束（惰性）
    conshdlr = TSPconshdlr(x,y,pre_idx) # set up conshdlr with all variables
    model.includeConshdlr(conshdlr, "PCTSP", "PCTSP subtour eliminator", needscons=False)

    # 定义目标函数，需要扣除后一块外板到前一块外板的惩罚
    obj = 0
    for i in N:
        for j in N:
            if (i != j) & (j != pre_idx):
                obj -= penalty[i][j]*x[i,j]

    model.setObjective(obj, "maximize")
    model.optimize()

    return model, x, y

def gap_score(w,gaptol):
    n = len(w)
    penalty = np.zeros((n,n)).tolist()
    for i in range(n):
        for j in range(n):
            width_gap = w[j]-w[i]
            if -gaptol <= width_gap <= gaptol:
                penalty[i][j] = (width_gap >= 0)*width_gap**2+0.01*(width_gap < 0)*width_gap**2
            else:
                penalty[i][j] = 1e10 #表示一个不可行的gap
    return penalty

def extract_order_data(excel_file_path):
    try:
        # 读取Excel文件，假设第一行是列名
        df = pd.read_excel(excel_file_path, header=0)
        
        # 获取所有属性名称(列名)
        attribute_names = df.columns.tolist()
        
        return df, attribute_names
    
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None, None
    
def data_keep(data,keep):
    # data是传入的DataFrame
    # attributes是需要的一些属性值
    data = data[keep]
    # print("成功保留以下属性:")
    # print(keep)
    return data

from collections import defaultdict

def get_cycle_nodes(edges, start_node):
    # 1. 构建邻接表
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    # 2. 检查起点是否存在
    if start_node not in graph:
        raise ValueError(f"起点 {start_node} 不在环中！")
    
    # 3. 遍历环，记录节点（不重复起点）
    cycle_nodes = []
    current = start_node
    while True:
        cycle_nodes.append(current)
        next_nodes = graph.get(current, [])
        if not next_nodes:
            raise ValueError("边集合不构成环！")
        next_node = next_nodes[0]  # 默认取第一个邻接点
        
        # 如果下一个点是起点，结束遍历（不重复添加）
        if next_node == start_node:
            break
        
        # 防止无限循环（如果边集合有问题）
        if next_node in cycle_nodes:
            raise ValueError("边集合中存在重复路径，无法构成简单环！")
        
        current = next_node
    
    return cycle_nodes


if __name__ == "__main__":
    file_path = "data1.xlsx"  
    
    order_data, attributes = extract_order_data(file_path)
    
    keep = ['存储厂别','材料实际宽度','材料重量','分选度代码','FIN_USER_NAME']

    order_data = data_keep(order_data,keep)

    g = order_data['材料重量']

    w = order_data['材料实际宽度']

    order_data1 = order_data[order_data['存储厂别']=='LZ3']
    order_data2 = order_data[order_data['存储厂别']!='LZ3']

    order_data1_outer = order_data1[(order_data1['分选度代码']>=4) & (order_data1['FIN_USER_NAME']!='特斯拉(上海)有限公司') & (order_data1['材料实际宽度']>=1500)]
    order_data1_tesla = order_data1[(order_data1['FIN_USER_NAME']=='特斯拉(上海)有限公司') & (order_data1['材料实际宽度']>=1500)]

    order_data2_outer = order_data2[(order_data2['分选度代码']>=4) & (order_data2['FIN_USER_NAME']!='特斯拉(上海)有限公司') & (order_data2['材料实际宽度']>=1500)]
    order_data2_tesla = order_data2[(order_data2['FIN_USER_NAME']=='特斯拉(上海)有限公司') & (order_data2['材料实际宽度']>=1500)]

    order_data_fake_inner = order_data[(order_data['FIN_USER_NAME']=='宝山钢铁股份有限公司制造管理部') & (order_data['分选度代码']<=3)]
    N_fake = order_data_fake_inner.index

    N1 = order_data1_outer.index.tolist()
    N1 = sorted(N1, key=lambda i: w[i], reverse=True)
    N2 = order_data1_tesla.index.tolist()
    N2 = sorted(N2, key=lambda i: w[i], reverse=True)

    N_used = []

    gtotal = 0
    pre = N1[0]
    node_full = []

    for i in range(4):
        pre_idx = N1[i]
        lat_idx = N1[i+1]
        pre_w = w[pre_idx]
        lat_w = w[lat_idx]

        order_data1_tesla_tmp = order_data1[(order_data1['FIN_USER_NAME']=='特斯拉(上海)有限公司') & (order_data1['材料实际宽度']<=pre_w) & (order_data1['材料实际宽度']>=lat_w)]
        N = order_data1_tesla_tmp.index.tolist()
        N = list(set(N) - set(N_used))
        N.append(pre_idx)
        N.append(lat_idx)

        model, x, y = my_GI_rule_scip(N, g, w, pre_idx, lat_idx, num = 8, width_gap=250)

        edges = []

        for (i,j) in x: 
            if (model.getVal(x[i,j]) > 0.5):
                edges.append((i,j))
                if (i != pre_idx) or (i != lat_idx):
                    N_used.append(i)
        

        objective_value = model.getObjVal()

        '''
        model.freeAll()  # 显式释放所有资源
        del model        # 确保对象被销毁
        '''

        print("Optimal tour:", edges)
        print("Optimal cost:", objective_value)

        node_full += get_cycle_nodes(edges,pre_idx)
        node_full.pop()

        results = ''
        
        for i in node_full:
            width, weight, user_name, user_cls = order_data.loc[i, '材料实际宽度'], order_data.loc[i, '材料重量'], order_data.loc[i, 'FIN_USER_NAME'], order_data.loc[i, '分选度代码']
            user_cls = '内板' if user_cls <= 3 else '外板'
            results += '({}, {}, {}, {})\n'.format(width, weight, user_name, user_cls)

    print(results)


        




