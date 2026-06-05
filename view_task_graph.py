#!/usr/bin/env python3
"""
查看 task_layer_graph.pkl 的内容
"""
import pickle
import networkx as nx
import sys
import json

def view_task_graph(pkl_path: str):
    """查看任务图的详细信息"""

    # 加载图
    try:
        with open(pkl_path, 'rb') as f:
            graph = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 文件不存在: {pkl_path}")
        return
    except Exception as e:
        print(f"错误: 无法加载文件: {e}")
        return

    print("=" * 60)
    print("Task Layer Graph 信息")
    print("=" * 60)

    # 基本统计
    print(f"\n📊 基本统计:")
    print(f"  节点数量: {graph.number_of_nodes()}")
    print(f"  边数量: {graph.number_of_edges()}")
    print(f"  图类型: {type(graph).__name__}")
    print(f"  是否连通: {nx.is_connected(graph)}")

    if graph.number_of_nodes() == 0:
        print("\n⚠️  图为空，没有节点")
        return

    # 节点信息
    print(f"\n📝 节点列表 (前10个):")
    for i, node in enumerate(list(graph.nodes())[:10]):
        node_data = graph.nodes[node]
        print(f"  {i+1}. {node}")
        if node_data:
            for key, value in node_data.items():
                # 截断长文本
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"     - {key}: {value}")

    if graph.number_of_nodes() > 10:
        print(f"  ... (共 {graph.number_of_nodes()} 个节点)")

    # 边信息
    print(f"\n🔗 边列表 (前20条):")
    edges = list(graph.edges(data=True))
    for i, (u, v, data) in enumerate(edges[:20]):
        similarity = data.get('similarity', data.get('weight', 'N/A'))
        print(f"  {i+1}. {u} <-> {v}")
        print(f"     相似度: {similarity}")

    if graph.number_of_edges() > 20:
        print(f"  ... (共 {graph.number_of_edges()} 条边)")

    # 度分布
    print(f"\n📈 度分布:")
    degrees = dict(graph.degree())
    if degrees:
        max_degree = max(degrees.values())
        min_degree = min(degrees.values())
        avg_degree = sum(degrees.values()) / len(degrees)
        print(f"  最大度: {max_degree}")
        print(f"  最小度: {min_degree}")
        print(f"  平均度: {avg_degree:.2f}")

        # 找出度最高的节点
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  度最高的5个节点:")
        for node, degree in top_nodes:
            print(f"    {node}: {degree}")

    # 相似度分布
    if graph.number_of_edges() > 0:
        print(f"\n📊 相似度分布:")
        similarities = []
        for u, v, data in graph.edges(data=True):
            sim = data.get('similarity', data.get('weight'))
            if sim is not None:
                similarities.append(sim)

        if similarities:
            print(f"  最大相似度: {max(similarities):.4f}")
            print(f"  最小相似度: {min(similarities):.4f}")
            print(f"  平均相似度: {sum(similarities)/len(similarities):.4f}")

    # 导出选项
    print(f"\n💾 可用操作:")
    print(f"  1. 导出为JSON: python3 {sys.argv[0]} {pkl_path} --json graph.json")
    print(f"  2. 导出节点列表: python3 {sys.argv[0]} {pkl_path} --nodes")
    print(f"  3. 导出边列表: python3 {sys.argv[0]} {pkl_path} --edges")

def export_to_json(graph, output_path: str):
    """导出为JSON格式"""
    data = nx.node_link_data(graph)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 已导出到: {output_path}")

def export_nodes(graph):
    """导出所有节点"""
    print("节点列表:")
    for node in graph.nodes():
        print(node)

def export_edges(graph):
    """导出所有边"""
    print("边列表 (格式: source,target,similarity):")
    for u, v, data in graph.edges(data=True):
        sim = data.get('similarity', data.get('weight', 'N/A'))
        print(f"{u},{v},{sim}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 view_task_graph.py <path_to_pkl>")
        print("示例: python3 view_task_graph.py .db/unknown/alfworld/autogen/g-memory/g-memory/task_layer_graph.pkl")
        sys.exit(1)

    pkl_path = sys.argv[1]

    # 加载图
    try:
        with open(pkl_path, 'rb') as f:
            graph = pickle.load(f)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 根据参数执行不同操作
    if len(sys.argv) > 2:
        if sys.argv[2] == '--json' and len(sys.argv) > 3:
            export_to_json(graph, sys.argv[3])
        elif sys.argv[2] == '--nodes':
            export_nodes(graph)
        elif sys.argv[2] == '--edges':
            export_edges(graph)
        else:
            print(f"未知参数: {sys.argv[2]}")
    else:
        view_task_graph(pkl_path)
