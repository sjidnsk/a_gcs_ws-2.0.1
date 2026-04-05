"""
HybridAStarGCSPlanner 测试脚本

测试 A*与GCS分层轨迹规划器的各种场景和配置。
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # tests/unit -> tests -> project_root
src_dir = os.path.join(project_root, 'src')
scripts_dir = os.path.join(src_dir, 'path_planner', 'scripts')

# 添加路径
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from C_space_pkg.se2 import SE2ConfigurationSpace, create_rectangle_robot
from A_pkg.A_star_fast_optimized import FastSE2AStarPlanner, PlannerConfig as AStarPlannerConfig
from hybrid_astar_gcs_planner import HybridAStarGCSPlanner
from planner_support import PlannerConfig


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
             gcs_strategy: str = 'standard', gcs_cost: str = 'lunar_standard'):
    """
    运行测试
    
    Args:
        scenario: 测试场景 ('basic', 'complex', 'narrow', 'u_turn', 's_curve', 'dynamic', 'multi_goal', 'parking')
        gcs_strategy: GCS策略预设 ('standard', 'high_risk', 'emergency', 'complex')
        gcs_cost: GCS成本预设 ('lunar_standard', 'lunar_high_risk', 'lunar_emergency', 
                               'lunar_complex', 'time_optimal', 'path_optimal', 
                               'energy_optimal', 'balanced', 'smooth')
    """
    print(f"\n{'='*60}")
    print(f"测试场景: {scenario}, IRIS模式: np")
    print(f"GCS策略: {gcs_strategy}, GCS成本: {gcs_cost}")
    print(f"{'='*60}")
    
    # 创建地图和配置空间
    if scenario == 'basic':
        map_size = 200
        start = (5.0, 5.0, 0.0)
        goal = (18.0, 18.0, np.pi/4)
        corridor_width = 3.0
    elif scenario == 'narrow':
        map_size = 250
        start = (3.5, 12.5, 0.0)
        goal = (21.0, 12.5, 0.0)
        corridor_width = 1.5
    elif scenario == 'complex':
        map_size = 500
        start = (10.0, 10.0, 0.0)
        goal = (45.0, 45.0, np.pi/4)
        corridor_width = 2.0
    elif scenario == 'u_turn':
        map_size = 200
        start = (16.0, 7.0, 0.0)
        goal = (16.0, 9.5, np.pi)
        corridor_width = 2.5
    elif scenario == 's_curve':
        map_size = 200
        start = (10.0, 10.0, 0.0)
        goal = (14.0, 10.0, 0.0)
        corridor_width = 2.5
    elif scenario == 'dynamic':
        map_size = 200
        start = (4.0, 4.0, 0.0)
        goal = (14.0, 14.0, np.pi/2)
        corridor_width = 2.0
    elif scenario == 'multi_goal':
        map_size = 200
        start = (5.0, 10.0, 0.0)
        goal = (15.0, 10.0, 0.0)
        corridor_width = 2.5
    elif scenario == 'parking':
        map_size = 200
        start = (10.0, 13.0, 0.0)
        goal = (10.0, 5.0, 0.0)
        corridor_width = 2.0
    
    obstacle_map = create_test_map(map_size, scenario)
    c_space = SE2ConfigurationSpace(obstacle_map, resolution=0.1)
    
    path = plan_path(c_space, start, goal)
    if not path:
        print("路径规划失败")
        return
    
    # 执行分解（使用预设配置）
    config = PlannerConfig(
        use_iris=True,
        corridor_width=corridor_width,

        # GCS策略和成本预设
        gcs_strategy_preset=gcs_strategy,
        gcs_cost_preset=gcs_cost,

        # 可视化
        enable_visualization=True,
        save_visualization=True,
        output_dir="./output",

        # 阿克曼车辆参数
        ackermann_wheelbase=2.5,
        ackermann_v_min=0.0,
        ackermann_v_max=5.0,
        ackermann_delta_min=-np.pi/4,
        ackermann_delta_max=np.pi/4,
    )
    
    planner = HybridAStarGCSPlanner(c_space, config)
    result = planner.process(path)
    
    # 打印结果
    print(f"\n结果: 区域数={result.num_obstacles}, 面积={result.obstacle_area:.2f}m², 时间={result.total_time:.2f}s")
    print(f"IRIS模式: {result.iris_mode_used}")
    print(f"GCS模式: {result.gcs_mode}")
    print(f"GCS策略: {gcs_strategy}, GCS成本: {gcs_cost}")


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'basic'
    gcs_strategy = sys.argv[2] if len(sys.argv) > 2 else 'standard'
    gcs_cost = sys.argv[3] if len(sys.argv) > 3 else 'lunar_standard'
    
    # 显示帮助
    if scenario in ['-h', '--help']:
        print("\n使用方法:")
        print("  python test_hybrid_astar_gcs_planner.py [scenario] [gcs_strategy] [gcs_cost]")
        print("\n参数说明:")
        print("  scenario: 测试场景")
        print("    - basic: 基础场景（200x200地图，默认）")
        print("    - narrow: 窄通道场景（250x250地图）")
        print("    - complex: 复杂地形场景（500x500地图，包含多个陨石坑、迷宫式障碍物）")
        print("    - u_turn: U型转弯场景（200x200地图，测试180度转弯）")
        print("    - s_curve: S型弯道场景（200x200地图，测试连续弯道）")
        print("    - dynamic: 动态障碍物场景（200x200地图，模拟移动障碍物）")
        print("    - multi_goal: 多目标点场景（200x200地图，测试经过多个waypoints）")
        print("    - parking: 泊车场景（200x200地图，模拟倒车入库）")
        print("\n  gcs_strategy: GCS策略预设")
        print("    - standard: 标准月面探索（默认）")
        print("    - high_risk: 高风险区域")
        print("    - emergency: 紧急避障")
        print("    - complex: 复杂地形")
        print("\n  gcs_cost: GCS成本预设")
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
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py basic standard lunar_standard")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py narrow standard lunar_standard")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py complex high_risk lunar_high_risk")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py basic emergency time_optimal")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py u_turn standard lunar_standard")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py s_curve balanced path_optimal")
        print("  python3 tests/unit/test_hybrid_astar_gcs_planner.py parking emergency smooth")
        sys.exit(0)
    
    # 运行测试
    run_test(scenario, gcs_strategy, gcs_cost)
