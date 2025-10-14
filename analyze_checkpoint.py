#!/usr/bin/env python3
"""
Checkpoint Analysis Script
Analyzes the structure of checkpoint files and compares with current model.
"""

import os
import torch
import yaml
from collections import defaultdict
from modelV1 import LatentDiffusion
from audiosr.latent_diffusion.util import instantiate_from_config

def analyze_checkpoint_keys(checkpoint_path):
    """分析检查点的键结构"""
    print(f"🔍 分析检查点: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print("❌ 检查点文件不存在")
        return

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # 获取state_dict
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("✅ 使用检查点中的 state_dict")
        elif "ema" in ckpt:
            state_dict = ckpt["ema"]
            print("✅ 使用检查点中的 EMA 权重")
        else:
            state_dict = ckpt
            print("✅ 使用检查点根级别权重")

        print(f"📊 检查点统计: {len(state_dict)} 个键")

        # 按组件分组分析
        components = defaultdict(list)
        for key in state_dict.keys():
            if "." in key:
                main_component = key.split(".")[0]
                components[main_component].append(key)
            else:
                components["root"].append(key)

        print("\n🏗️ 检查点组件结构:")
        for comp, keys in sorted(components.items()):
            print(f"  📦 {comp}: {len(keys)} 个键")
            if len(keys) <= 5:
                for key in keys:
                    print(f"    - {key}")
            else:
                for key in keys[:3]:
                    print(f"    - {key}")
                print(f"    ... 还有 {len(keys)-3} 个键")

        return state_dict

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def analyze_current_model():
    """分析当前模型结构"""
    print("\n🔍 分析当前模型结构:")

    try:
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 实例化模型
        model = instantiate_from_config(config['model'])
        model_state = model.state_dict()

        print(f"📊 当前模型统计: {len(model_state)} 个键")

        # 按组件分组分析
        components = defaultdict(list)
        for key in model_state.keys():
            if "." in key:
                main_component = key.split(".")[0]
                components[main_component].append(key)
            else:
                components["root"].append(key)

        print("\n🏗️ 当前模型组件结构:")
        for comp, keys in sorted(components.items()):
            print(f"  📦 {comp}: {len(keys)} 个键")
            if len(keys) <= 5:
                for key in keys:
                    print(f"    - {key}")
            else:
                for key in keys[:3]:
                    print(f"    - {key}")
                print(f"    ... 还有 {len(keys)-3} 个键")

        return model_state

    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        return None

def compare_structures(checkpoint_dict, model_dict):
    """比较检查点和模型结构"""
    print("\n🔍 结构对比分析:")

    if checkpoint_dict is None or model_dict is None:
        print("❌ 无法进行对比，缺少数据")
        return

    checkpoint_keys = set(checkpoint_dict.keys())
    model_keys = set(model_dict.keys())

    # 计算重叠和差异
    common_keys = checkpoint_keys & model_keys
    missing_in_checkpoint = model_keys - checkpoint_keys
    unexpected_in_checkpoint = checkpoint_keys - model_keys

    print(f"✅ 完全匹配的键: {len(common_keys)}")
    print(f"❌ 检查点中缺失的键: {len(missing_in_checkpoint)}")
    print(f"⚠️ 检查点中意外的键: {len(unexpected_in_checkpoint)}")

    if len(missing_in_checkpoint) > 0:
        print(f"\n❌ 缺失键示例 (前10个):")
        for key in list(missing_in_checkpoint)[:10]:
            print(f"  - {key}")

    if len(unexpected_in_checkpoint) > 0:
        print(f"\n⚠️ 意外键示例 (前10个):")
        for key in list(unexpected_in_checkpoint)[:10]:
            print(f"  - {key}")

    # 分析主要组件的匹配情况
    major_components = ["model", "first_stage_model", "cond_stage_models"]

    print(f"\n🔍 主要组件匹配分析:")
    for comp in major_components:
        comp_model_keys = {k for k in model_keys if k.startswith(comp + ".")}
        comp_checkpoint_keys = {k for k in checkpoint_keys if k.startswith(comp + ".")}

        if comp_model_keys and comp_checkpoint_keys:
            match_ratio = len(comp_model_keys & comp_checkpoint_keys) / len(comp_model_keys)
            status = "✅" if match_ratio > 0.8 else "⚠️" if match_ratio > 0.5 else "❌"
            print(f"  {status} {comp}: {match_ratio*100:.1f}% 匹配 ({len(comp_model_keys & comp_checkpoint_keys)}/{len(comp_model_keys)})")
        elif comp_model_keys:
            print(f"  ❌ {comp}: 检查点中完全缺失 (需要 {len(comp_model_keys)} 个键)")
        elif comp_checkpoint_keys:
            print(f"  ⚠️ {comp}: 仅在检查点中存在 ({len(comp_checkpoint_keys)} 个键)")

if __name__ == "__main__":
    # 获取检查点路径
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    checkpoint_path = config['train']['pretrained_path']

    print("=" * 60)
    print("🔬 检查点与模型结构分析工具")
    print("=" * 60)

    # 分析检查点
    checkpoint_dict = analyze_checkpoint_keys(checkpoint_path)

    # 分析当前模型
    model_dict = analyze_current_model()

    # 对比分析
    compare_structures(checkpoint_dict, model_dict)

    print("\n" + "=" * 60)
    print("✅ 分析完成!")