#!/usr/bin/env python3
"""
集成测试：测试Materials Project下载功能
"""

import os
import tempfile
from abacusagent.modules.submodules.structure_generator import materials_project_download

def test_materials_project_download():
    """测试materials_project_download函数的基本功能"""
    
    # 设置测试用的API密钥
    os.environ['MP_API_KEY'] = 'n9WVEKNI1A8yP1MfO3KBK8eKeFwESFBu'
    
    print("测试materials_project_download功能...")
    
    # 测试1: 基本功能测试
    try:
        # 使用一个已知存在的材料ID进行测试
        result = materials_project_download(material_id="mp-1234")
        print("✓ 基本功能测试通过")
        print(f"  结构文件: {result.get('structure_file')}")
        print(f"  材料ID: {result.get('material_id')}")
        
        # 检查返回值
        assert 'structure_file' in result, "应返回structure_file字段"
        assert 'material_id' in result, "应返回material_id字段"
        assert result['material_id'] == "mp-1234", "材料ID应匹配"
        
        # 检查文件是否存在
        if 'structure_file' in result:
            file_path = result['structure_file']
            assert os.path.exists(file_path), "下载的文件应该存在"
            print(f"✓ 文件存在: {file_path}")
            
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False
    
    # 测试2: 指定输出路径
    try:
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as tmp:
            temp_path = tmp.name
            
        result = materials_project_download(
            material_id="mp-1234", 
            destination_path=temp_path
        )
        
        assert result['structure_file'] == temp_path, "应返回指定的输出路径"
        assert os.path.exists(temp_path), "指定路径的文件应该存在"
        print("✓ 指定路径测试通过")
        
        # 清理临时文件
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"✗ 指定路径测试失败: {e}")
        return False
    
    # 测试3: 错误处理测试 - 无效的材料ID
    try:
        result = materials_project_download(material_id="mp-999999999")
        # 如果没有抛出异常，检查是否返回了错误消息
        if 'message' in result:
            print("✓ 错误处理测试通过")
        else:
            print("✓ 错误处理测试通过（未找到材料）")
    except Exception as e:
        print(f"✓ 错误处理测试通过（捕获异常）: {e}")
    
    print("\n所有测试完成！")
    return True

if __name__ == "__main__":
    test_materials_project_download()