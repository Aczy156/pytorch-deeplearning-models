import os


def get_file_list(source_dir):
    file_list = []
    """
    利用os进行扫描文件夹, os.walk()
        root: 给文件夹的根目录路径
        dirs: [list]存放当前目录下所有文件的名字
        files: [list]存放当前目录下所有文件 
    """
    # for root, dirs, files in os.walk(source_dir):

