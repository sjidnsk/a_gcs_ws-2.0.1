# rounding.py
# 路径舍入策略模块
# 该模块提供了多种从GCS（凸集图）松弛解中提取可行路径的舍入策略
# 在GCS问题中，松弛解通常给出分数解（边的流值在0-1之间），需要转换为整数解（完整路径）

import networkx as nx  # 用于图操作
import numpy as np    # 用于数值计算

# ====== 辅助函数 ======

def depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn):
    """
    深度优先搜索算法，用于从源点到目标点寻找路径
    
    参数:
        source: 起始顶点
        target: 目标顶点
        getCandidateEdgesFn: 获取候选边的函数，参数为(当前顶点, 已访问顶点列表)
        edgeSelectorFn: 选择边的函数，参数为候选边列表，返回(选中的边, 下一顶点)
    
    返回:
        path_edges: 从源点到目标点的边列表（路径）
    """
    visited_vertices = [source]  # 已访问的顶点列表
    path_vertices = [source]     # 当前路径的顶点列表
    path_edges = []             # 当前路径的边列表
    
    # 当路径的最后一个顶点不是目标点时继续搜索
    while path_vertices[-1] != target:
        # 获取当前顶点的候选边（排除已访问顶点）
        candidate_edges = getCandidateEdgesFn(path_vertices[-1], visited_vertices)
        
        # 如果没有候选边，回溯到上一个顶点
        if len(candidate_edges) == 0:
            path_vertices.pop()  # 移除最后一个顶点
            path_edges.pop()     # 移除最后一条边
        else:
            # 选择下一条边和下一个顶点
            next_edge, next_vertex = edgeSelectorFn(candidate_edges)
            visited_vertices.append(next_vertex)  # 标记为已访问
            path_vertices.append(next_vertex)     # 添加到路径
            path_edges.append(next_edge)          # 添加到路径
    
    return path_edges

def incomingEdges(gcs):
    """
    获取图中每个顶点的入边列表
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
    
    返回:
        incoming_edges: 字典，键为顶点ID，值为指向该顶点的边列表
    """
    # 初始化字典，键为顶点ID，值为空列表
    incoming_edges = {v.id(): [] for v in gcs.Vertices()}
    
    # 遍历所有边，将每条边添加到其终点顶点的入边列表中
    for e in gcs.Edges():
        incoming_edges[e.v().id()].append(e)
    
    return incoming_edges

def outgoingEdges(gcs):
    """
    获取图中每个顶点的出边列表
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
    
    返回:
        outgoing_edges: 字典，键为顶点ID，值为从该顶点出发的边列表
    """
    # 初始化字典，键为顶点ID，值为空列表
    outgoing_edges = {u.id(): [] for u in gcs.Vertices()}
    
    # 遍历所有边，将每条边添加到其起点顶点的出边列表中
    for e in gcs.Edges():
        outgoing_edges[e.u().id()].append(e)
    
    return outgoing_edges

def extractEdgeFlows(gcs, result):
    """
    从求解结果中提取每条边的流值
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
        result: 求解结果对象
    
    返回:
        flows: 字典，键为边ID，值为该边的流值(phi)
    """
    return {e.id(): result.GetSolution(e.phi()) for e in gcs.Edges()}

def greedyEdgeSelector(candidate_edges, flows):
    """
    贪婪选择策略：选择流值最大的边
    
    参数:
        candidate_edges: 候选边列表
        flows: 边ID到流值的映射字典
    
    返回:
        选中的边
    """
    # 获取每条候选边的流值
    candidate_flows = [flows[e.id()] for e in candidate_edges]
    # 选择流值最大的边
    return candidate_edges[np.argmax(candidate_flows)]

def randomEdgeSelector(candidate_edges, flows):
    """
    随机选择策略：按流值比例随机选择边
    
    参数:
        candidate_edges: 候选边列表
        flows: 边ID到流值的映射字典
    
    返回:
        选中的边
    """
    # 获取每条候选边的流值
    candidate_flows = np.array([flows[e.id()] for e in candidate_edges])
    # 计算选择每条边的概率（归一化流值）
    probabilities = candidate_flows / sum(candidate_flows)
    # 按概率随机选择一条边
    return np.random.choice(candidate_edges, p=probabilities)

# ====== 路径舍入策略 ======

def greedyForwardPathSearch(gcs, result, source, target, flow_tol=1e-5, **kwargs):
    """
    贪婪前向路径搜索：从源点开始，贪婪地选择流值最大的出边，直到到达目标点
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
        result: 求解结果对象
        source: 起始顶点
        target: 目标顶点
        flow_tol: 流值阈值，低于此值的边被视为0
    
    返回:
        paths: 包含一条路径的列表（即使只找到一条路径也返回列表）
    """
    # 获取每个顶点的出边
    outgoing_edges = outgoingEdges(gcs)
    # 提取边的流值
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        """
        获取当前顶点的候选边（未访问且流值大于阈值的出边）
        
        参数:
            current_vertex: 当前顶点
            visited_vertices: 已访问的顶点列表
        
        返回:
            候选边列表
        """
        # 过滤条件：目标顶点未访问且流值大于阈值
        keepEdge = lambda e: e.v() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in outgoing_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        """
        选择候选边中的最佳边
        
        参数:
            candidate_edges: 候选边列表
        
        返回:
            (选中的边, 下一顶点)
        """
        e = greedyEdgeSelector(candidate_edges, flows)
        return e, e.v()

    # 执行深度优先搜索，返回包含一条路径的列表
    return [depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn)]

def runTrials(source, target, getCandidateEdgesFn, edgeSelectorFn, max_paths=10, max_trials=1000):
    """
    运行多次路径搜索试验，收集多条不同的路径
    
    参数:
        source: 起始顶点
        target: 目标顶点
        getCandidateEdgesFn: 获取候选边的函数
        edgeSelectorFn: 选择边的函数
        max_paths: 最大路径数量
        max_trials: 最大尝试次数
    
    返回:
        paths: 找到的不同路径列表
    """
    paths = []    # 存储找到的路径
    trials = 0    # 尝试次数计数器
    
    # 当找到足够多的路径或达到最大尝试次数时停止
    while len(paths) < max_paths and trials < max_trials:
        trials += 1
        # 执行深度优先搜索
        path = depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn)
        # 如果找到新路径，添加到结果列表
        if path not in paths:
            paths.append(path)
    
    return paths

def randomForwardPathSearch(gcs, result, source, target, max_paths=10, max_trials=100, seed=None, flow_tol=1e-5, **kwargs):
    """
    随机前向路径搜索：从源点开始，按流值比例随机选择出边，多次尝试以找到多条路径
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
        result: 求解结果对象
        source: 起始顶点
        target: 目标顶点
        max_paths: 最大路径数量
        max_trials: 最大尝试次数
        seed: 随机数种子（可选）
        flow_tol: 流值阈值
    
    返回:
        paths: 找到的多条路径列表
    """
    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)

    # 获取每个顶点的出边
    outgoing_edges = outgoingEdges(gcs)
    # 提取边的流值
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        """
        获取当前顶点的候选边（未访问且流值大于阈值的出边）
        """
        keepEdge = lambda e: e.v() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in outgoing_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        """
        按流值比例随机选择边
        """
        e = randomEdgeSelector(candidate_edges, flows)
        return e, e.v()

    # 运行多次试验，收集不同路径
    return runTrials(source, target, getCandidateEdgesFn, edgeSelectorFn, max_paths, max_trials)

def greedyBackwardPathSearch(gcs, result, source, target, flow_tol=1e-5, **kwargs):
    """
    贪婪后向路径搜索：从目标点开始反向搜索，贪婪地选择流值最大的入边，直到到达源点
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
        result: 求解结果对象
        source: 起始顶点
        target: 目标顶点
        flow_tol: 流值阈值
    
    返回:
        paths: 包含一条路径的列表（路径顺序从源点到目标点）
    """
    # 获取每个顶点的入边
    incoming_edges = incomingEdges(gcs)
    # 提取边的流值
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        """
        获取当前顶点的候选边（未访问且流值大于阈值的入边）
        """
        keepEdge = lambda e: e.u() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in incoming_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        """
        选择流值最大的边
        """
        e = greedyEdgeSelector(candidate_edges, flows)
        return e, e.u()

    # 执行深度优先搜索（从目标点到源点），然后反转路径
    return [depthFirst(target, source, getCandidateEdgesFn, edgeSelectorFn)[::-1]]

def randomBackwardPathSearch(gcs, result, source, target, max_paths=10, max_trials=100, seed=None, flow_tol=1e-5, **kwargs):
    """
    随机后向路径搜索：从目标点开始反向搜索，按流值比例随机选择入边，多次尝试以找到多条路径
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
        result: 求解结果对象
        source: 起始顶点
        target: 目标顶点
        max_paths: 最大路径数量
        max_trials: 最大尝试次数
        seed: 随机数种子（可选）
        flow_tol: 流值阈值
    
    返回:
        paths: 找到的多条路径列表（每条路径顺序从源点到目标点）
    """
    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)

    # 获取每个顶点的入边
    incoming_edges = incomingEdges(gcs)
    # 提取边的流值
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        """
        获取当前顶点的候选边（未访问且流值大于阈值的入边）
        """
        keepEdge = lambda e: e.u() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in incoming_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        """
        按流值比例随机选择边
        """
        e = randomEdgeSelector(candidate_edges, flows)
        return e, e.u()

    # 运行多次试验（从目标点到源点），然后反转每条路径
    return [path[::-1] for path in runTrials(target, source, getCandidateEdgesFn, edgeSelectorFn, max_paths, max_trials)]

def MipPathExtraction(gcs, result, source, target, **kwargs):
    """
    MIP路径提取：使用混合整数规划方法提取路径（当前实现为贪婪前向搜索）
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
        result: 求解结果对象
        source: 起始顶点
        target: 目标顶点
    
    返回:
        paths: 包含一条路径的列表
    """
    return greedyForwardPathSearch(gcs, result, source, target)

def averageVertexPositionGcs(gcs, result, source, target, flow_min=1e-3, **kwargs):
    """
    基于平均顶点位置的路径提取：通过计算每个顶点的平均位置，然后在平均位置图上运行Dijkstra算法
    
    参数:
        gcs: Graph of Convex Sets (GCS) 对象
        result: 求解结果对象
        source: 起始顶点
        target: 目标顶点
        flow_min: 最小流值阈值，用于确定是否使用流加权平均
    
    返回:
        paths: 包含一条路径的列表
    """
    # 创建有向图
    G = nx.DiGraph()
    G.add_nodes_from(gcs.Vertices())

    # 初始化顶点数据字典，用于存储顶点位置和流值
    vertex_data = {}
    for v in gcs.Vertices():
        # 初始化为零向量，最后一个元素存储总流值
        vertex_data[v.id()] = np.zeros(v.set().ambient_dimension() + 1)

    # 计算每个顶点的平均位置
    for e in gcs.Edges():
        # 累加起点顶点的加权位置
        vertex_data[e.u().id()][:-1] += e.GetSolutionPhiXu(result)
        # 累加起点顶点的流值
        vertex_data[e.u().id()][-1] += result.GetSolution(e.phi())
        # 如果终点是目标点，也累加目标点的位置
        if e.v() == target:
            vertex_data[target.id()][:-1] += e.GetSolutionPhiXv(result)
            vertex_data[target.id()][-1] += result.GetSolution(e.phi())

    # 计算每个顶点的平均位置
    for v in gcs.Vertices():
        # 如果总流值大于阈值，计算加权平均位置
        if vertex_data[v.id()][-1] > flow_min:
            vertex_data[v.id()] = vertex_data[v.id()][:-1] / vertex_data[v.id()][-1]
        else:
            # 否则使用顶点集合的切比雪夫中心（最内切球中心）
            vertex_data[v.id()] = v.set().ChebyshevCenter()

    # 构建边的代价图
    for e in gcs.Edges():
        G.add_edge(e.u(), e.v())
        e_cost = 0
        # 计算边的总代价
        for cost in e.GetCosts():
            # 如果成本函数只依赖于起点顶点
            if len(cost.variables()) == e.u().set().ambient_dimension():
                e_cost += cost.evaluator().Eval(vertex_data[e.u().id()])
            # 如果成本函数依赖于起点和终点顶点
            elif len(cost.variables()) == e.u().set().ambient_dimension() + e.v().set().ambient_dimension():
                e_cost += cost.evaluator().Eval(np.append(vertex_data[e.u().id()], vertex_data[e.v().id()]))
            else:
                raise Exception("无法确定成本函数使用的变量。")
        
        # 设置边的长度属性（用于Dijkstra算法）
        G.edges[e.u(), e.v()]['l'] = np.squeeze(e_cost)
        
        # 检查边的长度是否为负（不应该出现）
        if G.edges[e.u(), e.v()]['l'] < 0:
            raise RuntimeError(f"边 {e} 的平均长度为负。考虑增加 flow_min。")

    # 使用Dijkstra算法在平均位置图上寻找最短路径
    path_vertices = nx.dijkstra_path(G, source, target, 'l')

    # 将顶点路径转换为边路径
    path_edges = []
    for u, v in zip(path_vertices[:-1], path_vertices[1:]):
        for e in gcs.Edges():
            if e.u() == u and e.v() == v:
                path_edges.append(e)
                break

    return [path_edges]