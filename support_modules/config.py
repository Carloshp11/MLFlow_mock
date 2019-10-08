from library.config import ConfigBase
from support_modules.misc import MixedParameterGrid


# noinspection PyAttributeOutsideInit
class MLFlowonfig(ConfigBase):
    def __init__(self, config_path: str, default_config_path=None, **kwargs):
        super().__init__(config_path, default_config_path, **kwargs)

    def _set_local_path(self, pathname):
        self.__setattr__(pathname, '{}/{}'.format(self.local_execution_path, self.__getattribute__(pathname).split('/')[-1]))

    def _update(self, attname, func):
        self.__setattr__(attname, func(self.__getattr__(attname)))

    def post_process_dinamic_params(self) -> None:
        # noinspection PyTypeChecker
        self.hyperparameters = MixedParameterGrid(self.hyperparameters)
