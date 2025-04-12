import os
import shutil

from . import common


class FolderEncoder:
    def __init__(self):
        pass
    
    @staticmethod
    def run(input_folder, output_folder, except_names=[], ignore_files=[]):
        # copy all files in input folder to output folder
        common.create_folder(output_folder)
        shutil.copytree(input_folder, output_folder, dirs_exist_ok=True)
        
        # get all path of .py files in output folder
        py_names, py_paths = common.get_items_from_folder(output_folder, ['.py'])
        
        ext_module_names = []
        for py_path in py_paths:
            folder_idx = py_path.find(output_folder)
            ext_module_name = py_path[folder_idx + len(output_folder) + 1:-3]
            ext_module_names.append([ext_module_name.replace('-', '_').replace('/', '.'), ext_module_name + '.py'])
        
            
        # create setup.py file
        """
        from distutils.core import setup
        from distutils.extension import Extension
        from Cython.Distutils import build_ext

        ext_modules = [
            Extension("app.sample_async",  ["app/sample_async.py"]),
        #   ... all your modules that need be compiled ...
        ]

        for ext_module in ext_modules:
            ext_module.cython_directives = {'language_level': "3"} #all are Python-3
            
        setup(
            name = 'async Engine',
            cmdclass = {'build_ext': build_ext},
            ext_modules = ext_modules
        )
        """
        
        script_lines = [
            "from distutils.core import setup",
            "from distutils.extension import Extension",
            "from Cython.Distutils import build_ext",
            "",
        ]
        
        script_lines.append("ext_modules = [")
        for ext_module_name, py_path in ext_module_names:
            script_lines.append(f"    Extension(\"{ext_module_name}\",  [\"{py_path}\"]),")
        script_lines.append("]")
        script_lines.append("")
        
        script_lines.append("for ext_module in ext_modules:")
        script_lines.append("    ext_module.cython_directives = {'language_level': \"3\"}")
        script_lines.append("")
        
        script_lines.append("setup(")
        script_lines.append("    name = '" + output_folder.split('/')[-1] + "',")
        script_lines.append("    cmdclass = {'build_ext': build_ext},")
        script_lines.append("    ext_modules = ext_modules")
        script_lines.append(")")
        
        script = "\n".join(script_lines)
            
        # write script to setup.py file
        with open(os.path.join(output_folder, 'setup.py'), 'w') as f:
            f.write(script)
            
        # run setup.py file
        os.system(f"cd {output_folder} && python setup.py build_ext --inplace")
        
        # remove all .py files
        print("Removing .py files...")
        for py_name, py_path in zip(py_names, py_paths):
            if py_name not in except_names:
                os.remove(py_path)
                
        # find all ignore files in output folder then remove them
        print("Removing ignore files...")
        for ignore_file in ignore_files:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    if file == ignore_file:
                        os.remove(os.path.join(root, file))
                
        print("Done!")