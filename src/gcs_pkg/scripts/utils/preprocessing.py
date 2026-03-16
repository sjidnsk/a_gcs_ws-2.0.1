import numpy as np
from pydrake.all import MathematicalProgram, Solve
from time import time

def removeRedundancies(gcs, s, t, tol=1e-4, verbose=False):
    """
    移除图中冗余边，优化图结构以提高后续计算效率
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象，表示凸集图
        s: 起始顶点 (source vertex)
        t: 目标顶点 (target vertex)
        tol: 容差值，默认为 1e-4，用于数值计算的精度控制
        verbose: 是否打印详细处理信息，默认为 False
    
    返回:
        preprocessing_times: 包含预处理各阶段耗时的字典
    """
    
    # 记录预处理总时间和求解线性规划的时间
    preprocessing_times = {'total': time(), 'linear_programs': 0}

    if verbose:
        print('预处理前的边数量:', len(gcs.Edges()))

    # 定义辅助函数：获取与顶点 w 相连的入边和出边索引
    # inedges_w: 返回所有指向顶点 w 的边的索引列表
    inedges_w = lambda w: [k for k, e in enumerate(gcs.Edges()) if e.v() == w]
    # outedges_w: 返回所有从顶点 w 出发的边的索引列表
    outedges_w = lambda w: [k for k, e in enumerate(gcs.Edges()) if e.u() == w]

    # 确保起点 s 没有入边，终点 t 没有出边
    removal_edges = []
    # 收集所有从 s 出发的入边和指向 t 的出边
    for k in inedges_w(s) + outedges_w(t):
        removal_edges.append(gcs.Edges()[k])
    # 将这些边标记为不可用（添加 phi 约束为 False）
    for e in removal_edges:
        e.AddPhiConstraint(False)

    # 获取图中边的总数
    nE = len(gcs.Edges())
    # 创建全零和全一的边向量，用于后续约束定义
    zeroE = np.zeros(nE)
    onesE = np.ones(nE)
    
    # 创建数学规划问题
    prog = MathematicalProgram()
    
    # 定义变量 f: 从起点 s 到各顶点 u 的流
    f = prog.NewContinuousVariables(nE, 'f')
    # 添加 f 的边界约束: 0 ≤ f ≤ 1
    f_limits = prog.AddBoundingBoxConstraint(zeroE, onesE, f).evaluator()

    # 定义变量 g: 从各顶点 v 到终点 t 的流
    g = prog.NewContinuousVariables(nE, 'g')
    # 添加 g 的边界约束: 0 ≤ g ≤ 1
    g_limits = prog.AddBoundingBoxConstraint(zeroE, onesE, g).evaluator()

    # 初始化约束容器
    nV = len(gcs.Vertices())
    conservation_f = [None] * nV  # f 的流量守恒约束
    conservation_g = [None] * nV  # g 的流量守恒约束
    degree = [None] * nV          # 度约束

    # 为每个顶点设置流量守恒约束
    for i, w in enumerate(gcs.Vertices()):
    
        # 获取与顶点 w 相连的入边和出边索引
        Ew_in = inedges_w(w)
        Ew_out = outedges_w(w)
        Ew = Ew_in + Ew_out
        fw = f[Ew]
        
        # 构建流量守恒矩阵: 入边系数为 +1, 出边系数为 -1
        A = np.hstack((np.ones((1, len(Ew_in))), - np.ones((1, len(Ew_out)))))
        
        # 起点 s 的特殊处理: 出流比入流多 1
        if w == s:
            conservation_f[i] = prog.AddLinearEqualityConstraint(A, [-1], fw).evaluator()
        # 其他顶点: 入流等于出流
        else:
            conservation_f[i] = prog.AddLinearEqualityConstraint(A, [0], fw).evaluator()

        # 对 g 流量做类似处理
        gw = g[Ew]
        # 终点 t 的特殊处理: 入流比出流多 1
        if w == t:
            conservation_g[i] = prog.AddLinearEqualityConstraint(A, [1], gw).evaluator()
        # 其他顶点: 入流等于出流
        else:
            conservation_g[i] = prog.AddLinearEqualityConstraint(A, [0], gw).evaluator()

        # 度约束: 对于有入边的顶点，确保 f 和 g 不会同时为正
        # (即一个顶点不能同时作为路径的中间点和终点/起点)
        if len(Ew_in) > 0:
            # 构建约束矩阵: f 和 g 的入边系数均为 1
            A = np.ones((1, 2 * len(Ew_in)))
            # 拼接 f 和 g 的入边变量
            fgin = np.concatenate((f[Ew_in], g[Ew_in]))
            # 添加约束: 0 ≤ f + g ≤ 1
            degree[i] = prog.AddLinearConstraint(A, [0], [1], fgin).evaluator()

    # 用于存储冗余边的列表
    redundant_edges = []
    
    # 检查每条边是否冗余
    for e in gcs.Edges():
        # 获取边的起点和终点在顶点列表中的索引
        i = gcs.Vertices().index(e.u())
        j = gcs.Vertices().index(e.v())

        # 临时修改约束以检查当前边 e = (u,v) 是否可能出现在路径中
        
        # 起点约束调整
        if s == e.u():
            # 如果当前边从起点出发，则禁止其他出边
            f_limits.set_bounds(zeroE, zeroE)
            conservation_f[i].set_bounds([0], [0])
        else:
            # 否则要求起点有单位出流
            conservation_f[i].set_bounds([1], [1])
        
        # 终点约束调整
        if t == e.v():
            # 如果当前边指向终点，则禁止其他入边
            g_limits.set_bounds(zeroE, zeroE)
            conservation_g[j].set_bounds([0], [0])
        else:
            # 否则要求终点有单位入流
            conservation_g[j].set_bounds([-1], [-1])

        # 禁止终点顶点有其他入边（针对当前检查的边）
        degree[j].set_bounds([0], [0])

        # 求解线性规划问题，检查是否存在可行流
        result = Solve(prog)
        # 记录求解时间
        preprocessing_times['linear_programs'] += result.get_solver_details().optimizer_time
        
        # 如果无可行解，则当前边 e 是冗余的
        if not result.is_success():
            redundant_edges.append(e)

        # 重置约束以检查下一条边
        
        # 重置起点约束
        if s == e.u():
            f_limits.set_bounds(zeroE, onesE)
            conservation_f[i].set_bounds([-1], [-1])
        else:
            conservation_f[i].set_bounds([0], [0])
        
        # 重置终点约束
        if t == e.v():
            g_limits.set_bounds(zeroE, onesE)
            conservation_g[j].set_bounds([1], [1])
        else:
            conservation_g[j].set_bounds([0], [0])
        
        # 重置度约束
        degree[j].set_bounds([0], [1])

    # 将所有冗余边标记为不可用
    for e in redundant_edges:
        e.AddPhiConstraint(False)

    # 计算总耗时
    preprocessing_times['total'] = time() - preprocessing_times['total']
    
    # 打印详细信息（如果启用）
    if verbose:
        print('预处理后的边数量:', len(gcs.Edges()) - len(redundant_edges) - len(removal_edges))
        print('预处理总耗时:', preprocessing_times['total'])
        print('求解线性规划耗时:', preprocessing_times['linear_programs'])

    return preprocessing_times