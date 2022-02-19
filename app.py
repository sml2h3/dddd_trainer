import fire

from loguru import logger
from utils import project_manager
from utils import cache_data
from utils import load_cache


class App:

    def __init__(self):
        logger.info("\nHello baby~")

    def create(self, project_name: str, single: bool = False):
        logger.info("\nCreate Project ----> {}".format(project_name))
        pm = project_manager.ProjectManager()
        pm.create_project(project_name, single)

    def cache(self, project_name: str, base_path: str, search_type: str = "name"):
        logger.info("\nCaching Data ----> {}\nPath ----> {}".format(project_name, base_path))
        cache = cache_data.CacheData(project_name)
        cache.cache(base_path, search_type)
        pass

    def test_load(self):
        load = load_cache.GetLoader("test1")
        val = load.loaders['val']
        val = iter(val)
        for inputs, labels, labels_length in val:
            print(inputs, labels, labels_length)



if __name__ == '__main__':
    fire.Fire(App)
