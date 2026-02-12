import os
import tempfile
import unittest
from pathlib import Path
from abacusagent.modules.submodules.structure_generator import materials_project_download

class TestMaterialsProjectDownload(unittest.TestCase):
    
    def setUp(self):
        self.api_key = os.environ.get('MP_API_KEY', 'test_key')
        os.environ['MP_API_KEY'] = self.api_key
        
    def test_materials_project_download_basic(self):
        result = materials_project_download(material_id="mp-1234")
        
        self.assertIn('structure_file', result)
        self.assertIn('material_id', result)
        self.assertEqual(result['material_id'], "mp-1234")
        
        # 清理下载的文件
        if 'structure_file' in result and os.path.exists(result['structure_file']):
            os.unlink(result['structure_file'])
        
    def test_materials_project_download_with_custom_path(self):
        """测试指定输出路径的功能"""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as tmp:
            temp_path = tmp.name
            
        try:
            result = materials_project_download(
                material_id="mp-1234", 
                destination_path=temp_path
            )
            
            self.assertEqual(result['structure_file'], temp_path)
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_materials_project_download_error_handling(self):
        result = materials_project_download(material_id="mp-999999999")
        
        self.assertIn('message', result)
        
        # 如果下载成功（虽然不应该），清理文件
        if 'structure_file' in result and os.path.exists(result.get('structure_file', '')):
            os.unlink(result['structure_file'])

if __name__ == '__main__':
    unittest.main()
