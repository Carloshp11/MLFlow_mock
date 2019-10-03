import os

from library.config import ConfigBase


# noinspection PyAttributeOutsideInit
class ChurnConfig(ConfigBase):
    def __init__(self, config_path: str, default_config_path=None, **kwargs):
        super().__init__(config_path, default_config_path, **kwargs)

    def _set_local_path(self, pathname):
        self.__setattr__(pathname, '{}/{}'.format(self.local_execution_path, self.__getattribute__(pathname).split('/')[-1]))

    def _update(self, attname, func):
        self.__setattr__(attname, func(self.__getattr__(attname)))

    def post_process_dinamic_params(self, args) -> None:
        # Here you may transform some paths in objects or perform some format changes
        env_bucket = self.environment_buckets[args.env]
        self.local_execution = os.path.basename(os.path.expanduser("~")) == 'hadoop'
        for path in ('actives_path',
                     'board_export_path',
                     'board_path',
                     'mainfeatures_path',
                     'metrics_path',
                     'registered_user_path'):
            self._update(path, lambda x: x.format(env_bucket))

        for path in ('threshold_path',
                     'predict_path'):
            self._update(path, lambda x: x.format(env_bucket, args.brand))

        self.model_path = self.model_path.format(self.environment_buckets['ai-' + args.env],
                                                 args.brand,
                                                 str(args.model_version))

        fallback_paths = [k for k in self.keys() if '_path' in k and k not in ('config_path', 'default_config_path', 'local_execution_path')]
        for path in fallback_paths:
            self.__setattr__(path + '_fallback', self.__getattribute__(path))
            if self.local_execution:
                self._set_local_path(path)

        self.brand = args.brand
        self.brand_id = self.brand_id_dict[args.brand]

    def config_compliance_checks(self):
        # You check here the integrity of the configurations loaded
        for k, v in [(k, getattr(self, k)) for k in self.keys() if isinstance(getattr(self, k), str)]:
            assert '{}' not in v, '{} has not been completelly processed ({})'.format(k, v)
