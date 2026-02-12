import os
import tempfile
import unittest
from pathlib import Path
from abacusagent.modules.submodules.structure_generator import materials_project_download

class TestMaterialsProjectDownload(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        # 从环境变量获取API密钥，如果没有则使用默认值用于测试
        self.api_key = os.environ.get('MP_API_KEY', 'n9WVEKNI1A8yP1MfO3KBK8eKeFwESFBu')
        os.environ['MP_API_KEY'] = self.api_key
        
    def test_materials_project_download_basic(self):
        """测试materials_project_download基本功能"""
        # 测试基本功能 - 使用一个合理的材料ID
        # 注意：由于API限制，我们主要测试函数接口和逻辑正确性
        result = materials_project_download(material_id="mp-1234")
        
        # 检查返回值结构
        self.assertIn('structure_file', result)
        self.assertIn('material_id', result)
        self.assertEqual(result['material_id'], "mp-1234")
        
    def test_materials_project_download_with_custom_path(self):
        """测试指定输出路径的功能"""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as tmp:
            temp_path = tmp.name
            
        try:
            result = materials_project_download(
                material_id="mp-1234", 
                destination_path=temp_path
            )
            
            # 检查是否返回了指定的路径
            self.assertEqual(result['structure_file'], temp_path)
            # 检查文件是否存在
            self.assertTrue(os.path.exists(temp_path))
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_materials_project_download_error_handling(self):
        """测试错误处理功能"""
        # 测试无效材料ID的情况
        result = materials_project_download(material_id="mp-999999999")
        
        # 应该返回错误信息而不是抛出异常
        self.assertIn('message', result)

if __name__ == '__main__':
    unittest.main()