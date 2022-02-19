import os
from configs import Config
from loguru import logger


class ProjectManager:

    def __init__(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects")

    def create_project(self, project_name: str, single: bool = False):
        project_base_path = os.path.join(self.base_path, project_name)
        logger.info("Creating Directory... ----> {}".format(project_base_path))
        if not os.path.exists(project_base_path):
            os.mkdir(project_base_path)
            if not os.path.exists(project_base_path):
                logger.error("Directory create failed! ----> {}".format(project_base_path))
                return False
            models_path = os.path.join(project_base_path, "models")
            logger.info("Creating Directory... ----> {}".format(models_path))
            os.mkdir(models_path)

            cache_path = os.path.join(project_base_path, "cache")
            logger.info("Creating Directory... ----> {}".format(cache_path))
            os.mkdir(cache_path)

            checkpoints_path = os.path.join(project_base_path, "checkpoints")
            logger.info("Creating Directory... ----> {}".format(checkpoints_path))
            os.mkdir(checkpoints_path)

            config_path = os.path.join(os.path.join(project_base_path, "config.yaml"))
            logger.info("Creating {} Config File... ----> {}".format("CNN" if single else "CRNN", config_path))
            conf = Config(project_name)
            conf.make_config(single=single)

            logger.info("Create Project Success! ----> {}".format(project_name))
        else:
            logger.error("Directory already exists! ----> {}".format(project_base_path))
            return False
