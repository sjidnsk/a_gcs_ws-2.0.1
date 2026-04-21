"""
HybridAStarGCSPlanner 测试脚本

测试 A*与GCS分层轨迹规划器的各种场景和配置。
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional

try:
    from pydrake.geometry.optimization import HPolyhedron
    from pydrake.trajectories import BsplineTrajectory
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    HPolyhedron = None
    BsplineTrajectory = None

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # tests/unit -> tests -> project_root
src_dir = os.path.join(project_root, 'src')
scripts_dir = os.path.join(src_dir, 'path_planner', 'scripts')

# 添加路径
# 移除空字符串(如果存在)
if '' in sys.path:
    sys.path.remove('')

# 确保src_dir在最前面,这样新的visualization包会被优先导入
if src_dir in sys.path:
    sys.path.remove(src_dir)
sys.path.insert(0, src_dir)

if scripts_dir not in sys.path:
    sys.path.insert(1, scripts_dir)

# 添加ackermann_gcs_pkg路径(在src_dir之后)
ackermann_pkg_dir = os.path.join(src_dir, 'ackermann_gcs_pkg')
if ackermann_pkg_dir in sys.path:
    sys.path.remove(ackermann_pkg_dir)
sys.path.insert(2, ackermann_pkg_dir)

from C_space_pkg.se2 import SE2ConfigurationSpace, create_rectangle_robot
from A_pkg.A_star_fast_optimized import FastSE2AStarPlanner, PlannerConfig as AStarPlannerConfig
from path_planner.scripts.hybrid_astar_gcs_planner import HybridAStarGCSPlanner
from path_planner.scripts.planner_support import PlannerConfig

# 导入 ackermann_gcs_pkg 模块
from ackermann_gcs_pkg.ackermann_gcs_planner import AckermannGCSPlanner
from ackermann_gcs_pkg.ackermann_data_structures import (
    VehicleParams,
    EndpointState,
    BezierConfig,
    PlanningResult
)

# 导入可视化模块
from visualization.ackermann import visualize_ackermann_gcs_enhanced


# ==================== 阿克曼GCS配置 ====================

# 默认车辆参数
DEFAULT_VEHICLE_PARAMS = VehicleParams(
    wheelbase=2.5,                    # 轴距（米）
    max_steering_angle=np.deg2rad(85),  # 最大转向角 85°（弧度）
    max_velocity=10.0,                # 最大速度（米/秒）
    max_acceleration=8.0              # 最大加速度（米/秒²）
)

# 场景配置字典
SCENARIO_CONFIGS = {
    'basic': {
        'map_size': 200,
        'start': (18.0, 2.25, 2.5),
        'goal': (18.0, 19.5, 0.1),
        'corridor_width': 100.0
    },
    'minimal': {
        'map_size': 100,
        'start': (5.0, 5.0, 0.0),
        'goal': (8.0, 5.0, 0.0),
        'corridor_width': 5.0
    },
    'simple_straight': {
        'map_size': 200,
        'start': (5.0, 10.0, 0.0),
        'goal': (15.0, 10.0, 0.0),
        'corridor_width': 4.0
    },
    'gentle_turn': {
        'map_size': 200,
        'start': (5.0, 10.0, 0.0),
        'goal': (15.0, 15.0, np.pi/6),
        'corridor_width': 4.0
    },
    'sharp_turn': {
        'map_size': 200,
        'start': (5.0, 10.0, 0.0),
        'goal': (15.0, 10.0, np.pi/3),
        'corridor_width': 5.0
    },
    'corridor_passage': {
        'map_size': 200,
        'start': (5.0, 10.0, 0.0),
        'goal': (25.0, 10.0, 0.0),
        'corridor_width': 4.0
    },
    'slalom': {
        'map_size': 250,
        'start': (5.0, 12.5, 0.0),
        'goal': (30.0, 12.8, 0.0),
        'corridor_width': 4.0
    },
    'maze_navigation': {
        'map_size': 300,
        'start': (30.0, 30.0, 0.0),
        'goal': (60.0, 60.0, np.pi/4),
        'corridor_width': 4.0
    },
    'narrow': {
        'map_size': 250,
        'start': (1.5, 12.5, -np.pi/2),
        'goal': (23.0, 13.0, np.pi/2),
        'corridor_width': 100.0
    },
    'complex': {
        'map_size': 500,
        'start': (10.0, 10.0, 0.0),
        'goal': (45.0, 45.0, np.pi/4),
        'corridor_width': 2.0
    },
    'u_turn': {
        'map_size': 200,
        'start': (16.0, 7.0, 0.0),
        'goal': (16.0, 9.5, np.pi),
        'corridor_width': 2.5
    },
    's_curve': {
        'map_size': 200,
        'start': (10.0, 10.0, 0.0),
        'goal': (14.0, 10.0, 0.0),
        'corridor_width': 2.5
    },
    'dynamic': {
        'map_size': 200,
        'start': (4.0, 4.0, 0.0),
        'goal': (14.0, 14.0, np.pi/2),
        'corridor_width': 2.0
    },
    'multi_goal': {
        'map_size': 200,
        'start': (5.0, 10.0, 0.0),
        'goal': (15.0, 10.0, 0.0),
        'corridor_width': 2.5
    },
    'parking': {
        'map_size': 200,
        'start': (10.0, 13.0, 0.0),
        'goal': (10.0, 5.0, 0.0),
        'corridor_width': 2.0
    }
}


# ==================== 阿克曼GCS辅助函数 ====================

def create_endpoint_state(position: Tuple[float, float],
                         heading: float,
                         velocity: Optional[float] = None) -> EndpointState:
    """
    创建端点状态

    Args:
        position: 位置 (x, y)
        heading: 航向角（弧度）
        velocity: 速度（米/秒），默认None（不约束速度）

    Returns:
        EndpointState对象
    """
    return EndpointState(
        position=np.array(position),
        heading=heading,
        velocity=velocity
    )


def convert_iris_to_hpolyhedron(iris_regions: List) -> List[HPolyhedron]:
    """
    将IRIS区域转换为HPolyhedron

    Args:
        iris_regions: IRIS区域列表(支持iris_np和iriszo)

    Returns:
        HPolyhedron列表
    """
    workspace_regions = []
    for region in iris_regions:
        # 检查是否是iriszo的区域(有polyhedron属性)
        if hasattr(region, 'polyhedron'):
            # iriszo区域:直接使用polyhedron
            hpolyhedron = region.polyhedron
        elif hasattr(region, 'A') and hasattr(region, 'b'):
            # iris_np区域:从A和b构造HPolyhedron
            A = region.A
            b = region.b
            hpolyhedron = HPolyhedron(A, b)
        else:
            raise ValueError(f"未知的IRIS区域类型: {type(region)}")
        workspace_regions.append(hpolyhedron)
    return workspace_regions


def print_ackermann_result(result: PlanningResult):
    """
    打印阿克曼GCS规划结果

    Args:
        result: 规划结果
    """
    print(f"\n{'='*60}")
    print(f"阿克曼GCS规划结果")
    print(f"{'='*60}")

    if result.success:
        print(f"✓ 规划成功")
        print(f"  求解时间: {result.solve_time:.2f}s")
        print(f"  SCP迭代次数: {result.num_iterations}")
        print(f"  收敛原因: {result.convergence_reason}")

        if result.trajectory_report:
            report = result.trajectory_report
            print(f"\n轨迹可行性: {'✓ 可行' if report.is_feasible else '✗ 不可行'}")

            if report.velocity_violation:
                v = report.velocity_violation
                print(f"  速度约束: {'✓ 满足' if not v.is_violated else f'✗ 违反 (max: {v.max_violation:.6f})'}")

            if report.acceleration_violation:
                a = report.acceleration_violation
                print(f"  加速度约束: {'✓ 满足' if not a.is_violated else f'✗ 违反 (max: {a.max_violation:.6f})'}")

            if report.curvature_violation:
                c = report.curvature_violation
                print(f"  曲率约束: {'✓ 满足' if not c.is_violated else f'✗ 违反 (max: {c.max_violation:.6f})'}")

            if report.workspace_violation:
                w = report.workspace_violation
                print(f"  工作空间约束: {'✓ 满足' if not w.is_violated else f'✗ 违反 (max: {w.max_violation:.6f})'}")
    else:
        print(f"✗ 规划失败")
        print(f"  求解时间: {result.solve_time:.2f}s")
        print(f"  错误信息: {result.error_message}")

    print(f"{'='*60}\n")


def visualize_ackermann_trajectory(trajectory: BsplineTrajectory,
                                  source: EndpointState,
                                  target: EndpointState,
                                  workspace_regions: List[HPolyhedron],
                                  vehicle_params: VehicleParams,
                                  output_path: str):
    """
    可视化阿克曼GCS轨迹

    Args:
        trajectory: 轨迹
        source: 起点状态
        target: 终点状态
        workspace_regions: 工作空间区域
        vehicle_params: 车辆参数
        output_path: 输出路径
    """
    visualize_ackermann_gcs_enhanced(
        trajectory=trajectory,
        vehicle_params=vehicle_params,
        workspace_regions=workspace_regions,
        source=source,
        target=target,
        save_path=output_path
    )

    print(f"可视化已保存到: {output_path}")


def run_ackermann_gcs_test(scenario: str,
                          obstacle_map: np.ndarray,
                          start: Tuple,
                          goal: Tuple,
                          corridor_width: float):
    """
    运行阿克曼GCS测试

    Args:
        scenario: 测试场景名称
        obstacle_map: 障碍物地图
        start: 起点 (x, y, theta)
        goal: 终点 (x, y, theta)
        corridor_width: 通道宽度
    """
    print(f"\n{'='*60}")
    print(f"测试场景: {scenario}, 模式: ackermann_gcs")
    print(f"{'='*60}")

    try:
        # 步骤1：创建配置空间
        c_space = SE2ConfigurationSpace(obstacle_map, resolution=0.1)

        # 步骤2：A*路径规划（用于生成IRIS区域）
        path = plan_path(c_space, start, goal)
        if not path:
            raise ValueError("A*路径规划失败")

        # 步骤3：执行IRIS分解
        config = PlannerConfig(
            use_iris=True,
            corridor_width=corridor_width,
            enable_visualization=False,
            save_visualization=False,
            enable_gcs_optimization=False  # 禁用GCS优化，只需要IRIS区域
        )
        planner = HybridAStarGCSPlanner(c_space, config)
        result = planner.process(path)

        if result.num_obstacles == 0:
            raise ValueError("IRIS分解失败，未生成可行区域")

        # 步骤4：转换IRIS区域为HPolyhedron
        workspace_regions = convert_iris_to_hpolyhedron(result.iris_np_result.regions)

        if not workspace_regions:
            raise ValueError("IRIS区域转换失败")

        # 步骤5：配置车辆参数
        vehicle_params = DEFAULT_VEHICLE_PARAMS

        # 步骤6：创建起终点状态
        source = create_endpoint_state(start[:2], start[2])
        target = create_endpoint_state(goal[:2], goal[2])

        # 步骤7：初始化AckermannGCSPlanner
        ackermann_planner = AckermannGCSPlanner(
            vehicle_params=vehicle_params,
            bezier_config=BezierConfig(order=5, continuity=1)
        )

        # 步骤8：执行轨迹规划
        planning_result = ackermann_planner.plan_trajectory(
            source=source,
            target=target,
            workspace_regions=workspace_regions,
            constraints=None,
            cost_weights={"time": 1.0, "path_length": 0.1, "energy": 0.01},
            verbose=True
        )

        # 步骤9：打印结果
        print_ackermann_result(planning_result)

        # 步骤10：可视化
        if planning_result.success:
            try:
                # 使用新的模块化可视化系统
                from visualization.ackermann import visualize_ackermann_gcs_enhanced, VisualizationConfig
                
                # 创建配置
                viz_config = VisualizationConfig(
                    num_samples=200,
                    show_iris_regions=True,
                    show_obstacles=True,
                    show_corridor=True,
                    show_astar_path=True,
                    show_3d_trajectory=True,
                    show_theta_profile=True,
                    figsize=(20, 14),
                    dpi=150
                )
                
                # 执行可视化
                visualize_ackermann_gcs_enhanced(
                    trajectory=planning_result.trajectory,
                    vehicle_params=vehicle_params,
                    workspace_regions=workspace_regions,
                    source=source,
                    target=target,
                    obstacle_map=obstacle_map,
                    astar_path=None,  # TODO: 从planner获取A*路径
                    corridor_width=corridor_width,
                    resolution=0.1,  # C_space的分辨率
                    save_path=f"./output/ackermann_gcs_{scenario}_enhanced.png",
                    config=viz_config
                )
                print(f"增强版可视化已保存到: ./output/ackermann_gcs_{scenario}_enhanced.png")
                
            except Exception as e:
                print(f"增强版可视化失败: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 回退到简化可视化
                try:
                    visualize_ackermann_gcs_enhanced(
                        trajectory=planning_result.trajectory,
                        vehicle_params=vehicle_params,
                        source=source,
                        target=target,
                        save_path=f"./output/ackermann_gcs_{scenario}.png",
                    )
                    print(f"简化可视化已保存到: ./output/ackermann_gcs_{scenario}.png")
                except Exception as e2:
                    print(f"简化可视化也失败: {str(e2)}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ 测试失败: {scenario}")
        print(f"  错误: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()


def create_test_map(map_size: int = 200, scenario: str = 'basic') -> np.ndarray:
    """创建测试地图"""
    obstacle_map = np.zeros((map_size, map_size), dtype=np.uint8)
    
    if scenario == 'basic':
        obstacle_map[40:80, 60:100] = 1
        obstacle_map[120:160, 40:80] = 1
        obstacle_map[100:140, 120:160] = 1
        obstacle_map[60:100, 140:160] = 1
        obstacle_map[80:100, 160:200] = 1
        for i in range(map_size):
            for j in range(map_size):
                if (i - 150)**2 + (j - 150)**2 < 25**2:
                    obstacle_map[i, j] = 1
    
    elif scenario == 'minimal':
        # 极简场景：完全开阔的空间，只有边界
        # 100x100地图，起点(5,5)到终点(8,5)，直线距离3米
        obstacle_map[0:10, :] = 1  # 上边界
        obstacle_map[90:100, :] = 1  # 下边界
        obstacle_map[:, 0:10] = 1  # 左边界
        obstacle_map[:, 90:100] = 1  # 右边界
    
    elif scenario == 'simple_straight':
        # 简单直线场景：开阔空间，少量障碍物
        # 在路径上方和下方放置一些障碍物，确保有足够的通行空间
        obstacle_map[30:50, 80:120] = 1  # 上方障碍物
        obstacle_map[150:170, 80:120] = 1  # 下方障碍物
        # 边界障碍物
        obstacle_map[0:20, :] = 1
        obstacle_map[180:200, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 180:200] = 1
    
    elif scenario == 'sharp_turn':
        # 急转弯场景：需要车辆进行较大角度的转弯
        # 创建一个L形通道
        obstacle_map[0:80, 0:80] = 1  # 左上角障碍物
        obstacle_map[120:200, 120:200] = 1  # 右下角障碍物
        # 边界
        obstacle_map[0:20, :] = 1
        obstacle_map[180:200, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 180:200] = 1
    
    elif scenario == 'corridor_passage':
        # 走廊通道场景：需要穿过障碍物之间的通道
        # 创建简单的障碍物布局
        obstacle_map[40:80, 70:90] = 1  # 上方障碍物
        obstacle_map[120:160, 70:90] = 1  # 下方障碍物
        # 边界
        obstacle_map[0:20, :] = 1
        obstacle_map[180:200, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 180:200] = 1
    
    elif scenario == 'slalom':
        # 绕桩场景：需要车辆绕过几个障碍物
        # 创建简单的交错障碍物
        obstacle_map[60:90, 80:110] = 1  # 第一个障碍物
        obstacle_map[120:150, 130:160] = 1  # 第二个障碍物
        obstacle_map[180:210, 80:110] = 1  # 第三个障碍物
        # 边界
        obstacle_map[0:20, :] = 1
        obstacle_map[230:250, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 230:250] = 1
    
    elif scenario == 'maze_navigation':
        # 迷宫导航场景：中等复杂度的障碍物布局
        # 外墙
        obstacle_map[0:20, :] = 1
        obstacle_map[280:300, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 280:300] = 1
        
        # 简单的内部障碍物
        obstacle_map[80:120, 80:120] = 1
        obstacle_map[160:200, 160:200] = 1
        obstacle_map[80:120, 180:220] = 1
    
    elif scenario == 'gentle_turn':
        # 温和转弯场景：提供足够的转弯空间
        # 在转弯路径外侧放置障碍物
        obstacle_map[30:60, 130:180] = 1  # 右上角障碍物
        obstacle_map[150:180, 30:80] = 1  # 左下角障碍物
        # 边界障碍物
        obstacle_map[0:20, :] = 1
        obstacle_map[180:200, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 180:200] = 1
    
    elif scenario == 'narrow':
        obstacle_map[0:100, :] = 1
        obstacle_map[150:250, :] = 1
    
    elif scenario == 'complex':
        # 复杂地形：多个不规则障碍物、窄通道、迷宫式布局
        # 1. 大型矩形障碍物群
        obstacle_map[50:150, 30:80] = 1
        obstacle_map[180:280, 120:170] = 1
        obstacle_map[300:400, 50:100] = 1
        obstacle_map[100:200, 350:400] = 1
        obstacle_map[350:450, 300:350] = 1
        
        # # 2. 窄通道障碍物
        # obstacle_map[250:350, 150:200] = 1
        # obstacle_map[250:350, 250:300] = 1
        
        # 3. 对角线障碍物
        for i in range(50, 150):
            for j in range(200, 300):
                if abs(i - j + 150) < 10:
                    obstacle_map[i, j] = 1
        
        # 4. 多个圆形障碍物（模拟陨石坑）
        circles = [
            (100, 250, 30),
            (200, 200, 25),
            (300, 350, 35),
            (400, 150, 28),
            (150, 400, 32),
            (380, 280, 22),
            (250, 100, 27),
            (80, 320, 20),
            (420, 380, 25),
            (320, 80, 23)
        ]
        for cx, cy, radius in circles:
            for i in range(max(0, cx-radius), min(map_size, cx+radius)):
                for j in range(max(0, cy-radius), min(map_size, cy+radius)):
                    if (i - cx)**2 + (j - cy)**2 < radius**2:
                        obstacle_map[i, j] = 1
        
        # 5. 迷宫式障碍物
        obstacle_map[200:250, 50:80] = 1
        obstacle_map[200:250, 100:130] = 1
        obstacle_map[200:250, 150:180] = 1
        obstacle_map[200:250, 200:230] = 1
        obstacle_map[200:250, 250:280] = 1
        obstacle_map[200:250, 300:330] = 1
        obstacle_map[200:250, 350:380] = 1
        obstacle_map[200:250, 400:430] = 1
        
        # 6. L形障碍物
        obstacle_map[50:100, 200:250] = 1
        obstacle_map[50:150, 200:220] = 1
        
        obstacle_map[300:350, 400:450] = 1
        obstacle_map[300:400, 400:420] = 1
        
        # 7. 分散的小障碍物
        small_obstacles = [
            (120, 120, 15), (180, 320, 12), (280, 220, 14),
            (350, 180, 13), (420, 320, 16), (90, 380, 11),
            (220, 380, 14), (380, 120, 12), (150, 80, 15),
            (320, 320, 13)
        ]
        for cx, cy, radius in small_obstacles:
            for i in range(max(0, cx-radius), min(map_size, cx+radius)):
                for j in range(max(0, cy-radius), min(map_size, cy+radius)):
                    if (i - cx)**2 + (j - cy)**2 < radius**2:
                        obstacle_map[i, j] = 1
    
    elif scenario == 'u_turn':
        # U型转弯场景：需要机器人进行180度转弯
        obstacle_map[0:120, 0:150] = 1  # 左侧墙
        obstacle_map[120:200, 0:45] = 1  # 上方墙
        obstacle_map[120:200, 105:150] = 1  # 下方墙
        # 在右侧留出通道（y=45-105），机器人从右侧进入U型区域
        
    elif scenario == 's_curve':
        # S型弯道场景：测试机器人在连续弯道中的路径规划
        obstacle_map[0:80, 0:150] = 1
        obstacle_map[120:200, 50:200] = 1
        obstacle_map[0:80, 150:200] = 1
        obstacle_map[120:200, 0:50] = 1
        
    elif scenario == 'dynamic':
        # 动态障碍物场景：模拟移动的障碍物
        obstacle_map[50:100, 50:100] = 1
        obstacle_map[150:200, 100:150] = 1
        obstacle_map[100:150, 150:200] = 1
        
    elif scenario == 'multi_goal':
        # 多目标点场景：测试经过多个 waypoints
        obstacle_map[50:80, 80:120] = 1
        obstacle_map[120:150, 50:80] = 1
        obstacle_map[120:150, 120:150] = 1
        obstacle_map[80:120, 170:200] = 1
        
    elif scenario == 'parking':
        # 泊车场景：模拟倒车入库
        # 创建一个U型停车位，机器人需要倒车进入
        obstacle_map[0:40, 0:200] = 1  # 左墙
        obstacle_map[160:200, 0:200] = 1  # 右墙
        obstacle_map[0:200, 0:40] = 1  # 顶部墙
        obstacle_map[0:200, 160:200] = 1  # 底部墙
        # 停车位隔断（缩小范围，避免与起点碰撞）
        obstacle_map[70:130, 70:90] = 1
    
    return obstacle_map


def plan_path(c_space: SE2ConfigurationSpace, start: Tuple, goal: Tuple) -> Optional[List]:
    """A*路径规划"""
    robot = create_rectangle_robot(length=1.5, width=1.0)
    planner = FastSE2AStarPlanner(
        c_space=c_space, robot=robot, min_radius=1.5,
        resolution=0.5, theta_resolution=16,
        config=AStarPlannerConfig(max_iterations=100000, goal_tolerance=0.5)
    )
    return planner.plan(start, goal)


def run_test(scenario: str = 'basic',
             mode: str = 'hybrid_astar_gcs',
             gcs_strategy: str = 'standard',
             gcs_cost: str = 'lunar_standard'):
    """
    运行测试

    Args:
        scenario: 测试场景
        mode: 规划模式
            - 'hybrid_astar_gcs': HybridAStar + GCS（默认）
            - 'ackermann_gcs': 阿克曼转向GCS
        gcs_strategy: GCS策略预设（仅hybrid_astar_gcs模式有效）
        gcs_cost: GCS成本预设（仅hybrid_astar_gcs模式有效）
    """
    # 获取场景配置
    config = SCENARIO_CONFIGS.get(scenario)
    if not config:
        raise ValueError(f"未知场景: {scenario}")

    # 创建地图
    obstacle_map = create_test_map(config['map_size'], scenario)

    # 根据模式选择测试函数
    if mode == 'ackermann_gcs':
        run_ackermann_gcs_test(
            scenario=scenario,
            obstacle_map=obstacle_map,
            start=config['start'],
            goal=config['goal'],
            corridor_width=config['corridor_width']
        )
    else:
        # 原有HybridAStarGCS测试流程
        print(f"\n{'='*60}")
        print(f"测试场景: {scenario}, IRIS模式: np")
        print(f"GCS策略: {gcs_strategy}, GCS成本: {gcs_cost}")
        print(f"{'='*60}")

        c_space = SE2ConfigurationSpace(obstacle_map, resolution=0.1)

        path = plan_path(c_space, config['start'], config['goal'])
        if not path:
            print("路径规划失败")
            return

        # 执行分解（使用预设配置）
        planner_config = PlannerConfig(
            use_iris=True,
            corridor_width=config['corridor_width'],

            # GCS策略和成本预设
            gcs_strategy_preset=gcs_strategy,
            gcs_cost_preset=gcs_cost,

            # 可视化
            enable_visualization=True,
            save_visualization=True,
            output_dir="./output"
        )

        planner = HybridAStarGCSPlanner(c_space, planner_config)
        result = planner.process(path)

        # 打印结果
        print(f"\n结果: 区域数={result.num_obstacles}, 面积={result.obstacle_area:.2f}m², 时间={result.total_time:.2f}s")
        print(f"IRIS模式: {result.iris_mode_used}")
        print(f"GCS策略: {gcs_strategy}, GCS成本: {gcs_cost}")


if __name__ == "__main__":
    import sys

    # 解析命令行参数
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'basic'
    mode = sys.argv[2] if len(sys.argv) > 2 else 'hybrid_astar_gcs'
    gcs_strategy = sys.argv[3] if len(sys.argv) > 3 else 'standard'
    gcs_cost = sys.argv[4] if len(sys.argv) > 4 else 'lunar_standard'

    # 显示帮助
    if scenario in ['-h', '--help']:
        print("\n使用方法:")
        print("  python test_hybrid_astar_gcs_planner.py [scenario] [mode] [gcs_strategy] [gcs_cost]")
        print("\n参数说明:")
        print("  scenario: 测试场景")
        print("    - basic: 基础场景（200x200地图，默认）")
        print("    - minimal: 极简场景（100x100地图，3米直线，推荐用于调试）")
        print("    - simple_straight: 简单直线场景（200x200地图）")
        print("    - gentle_turn: 温和转弯场景（200x200地图，30度转弯）")
        print("    - sharp_turn: 急转弯场景（200x200地图，60度转弯）")
        print("    - corridor_passage: 走廊通道场景（200x200地图，多个障碍物）")
        print("    - slalom: 绕桩场景（250x250地图，连续绕障）")
        print("    - maze_navigation: 迷宫导航场景（300x300地图，复杂布局）")
        print("    - narrow: 窄通道场景（250x250地图）")
        print("    - complex: 复杂地形场景（500x500地图）")
        print("    - u_turn: U型转弯场景（200x200地图）")
        print("    - s_curve: S型弯道场景（200x200地图）")
        print("    - dynamic: 动态障碍物场景（200x200地图）")
        print("    - multi_goal: 多目标点场景（200x200地图）")
        print("    - parking: 泊车场景（200x200地图）")
        print("\n  mode: 规划模式")
        print("    - hybrid_astar_gcs: HybridAStar + GCS（默认）")
        print("    - ackermann_gcs: 阿克曼转向GCS")
        print("\n  gcs_strategy: GCS策略预设（仅hybrid_astar_gcs模式有效）")
        print("    - standard: 标准月面探索（默认）")
        print("    - high_risk: 高风险区域")
        print("    - emergency: 紧急避障")
        print("    - complex: 复杂地形")
        print("\n  gcs_cost: GCS成本预设（仅hybrid_astar_gcs模式有效）")
        print("    - lunar_standard: 月面标准（默认）")
        print("    - lunar_high_risk: 月面高风险")
        print("    - lunar_emergency: 月面紧急")
        print("    - lunar_complex: 月面复杂")
        print("    - time_optimal: 时间优先")
        print("    - path_optimal: 路径优先")
        print("    - energy_optimal: 能量优先")
        print("    - balanced: 平衡策略")
        print("    - smooth: 高平滑性")
        print("\n示例:")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py basic hybrid_astar_gcs standard lunar_standard")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py basic ackermann_gcs")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py narrow ackermann_gcs")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py u_turn ackermann_gcs")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py parking ackermann_gcs")
        sys.exit(0)

    # 运行测试
    run_test(scenario, mode, gcs_strategy, gcs_cost)
