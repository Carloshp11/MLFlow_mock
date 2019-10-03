from os import path
import datetime

from library.code_patterns import AttDict
from library.io import load_yaml, pretty_dict


class ConfigBase(AttDict):
    """
    Base class for the config object.
    It must be innerited by the custom config object to receive several benefits:
        1) Native handling of base and default configurations, defined by their paths.
        2) Inneritance of the AttDict behaviour for improved dictionary interface.
        3) Support for automatic execution of post processingand compliance checks routines.
        4) PrettyPrint native method.
    """

    def __init__(self, config_path: str, default_config_path: str = None, add_extra_attributes: bool = True, **kwargs):
        """
        Initialization of the object.

        Base and default configurations are merged, giving prevalence to base attributes whenever they exist, but
        adding those attributes in the default configuration file when not explicitly declared on the base configuration.

        date, config_path and default_config_path attributes are automatically added to the object.
        :param config_path: Path to the configuration file.
        :param default_config_path: Optional. Path to the default configuration file.
        :param add_extra_attributes: Prevent the addition of date, config_path and default_config_path attributes to
                                     the object.
        """
        assert path.basename(config_path)[-5:] == '.yaml', 'config_path does\'t lead to a .yaml file.\n' \
                                                           'config_path: {}'.format(config_path)
        base_configuration = load_yaml(config_path)

        if default_config_path is not None:
            assert path.basename(default_config_path)[-5:] == '.yaml', 'default_config_path does\'t lead to a .yaml ' \
                                                                       'file.\n' \
                                                                       'default_config_path: {}'.format(
                default_config_path)
            default_configuration = load_yaml(default_config_path)
            base_configuration = self.__coalesce_dicts__(base_configuration, default_configuration)

        if add_extra_attributes:
            base_configuration = self.__coalesce_dict_keys__(base_configuration, 'date', datetime.datetime.now())
            base_configuration = self.__coalesce_dict_keys__(base_configuration, 'config_path', config_path)
            base_configuration = self.__coalesce_dict_keys__(base_configuration, 'default_config_path',
                                                             default_config_path)

        super().__init__(base_configuration, **kwargs)

    def print(self):
        pretty_dict(self)

    @staticmethod
    def __coalesce_dict_keys__(dict_, k, v):
        if k not in dict_.keys():
            dict_[k] = v
        return dict_

    @staticmethod
    def __coalesce_dicts__(dict_left, dict_right):
        for key in [_ for _ in dict_right.keys() if _ not in dict_left.keys()]:
            dict_left[key] = dict_right[key]
        return dict_left


if __name__ == '__main':

    # Example of a custom configuration file structure
    class ExampleSimpleCustomConfig(ConfigBase):
        def __init__(self, config_path: str, default_config_path=None, **kwargs):
            super().__init__(config_path, default_config_path, **kwargs)

        def post_process_dinamic_params(self):
            # Maybe transform some paths in objects or perform some format changes
            pass

        def config_compliance_checks(self):
            # You check here the integrity of the configurations loaded
            pass


    my_simple_project_configuration = ExampleSimpleCustomConfig('path_to_yaml.yaml')

    # Example of a complex custom configuration file

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    class ExampleCustomConfig(ConfigBase):
        def __init__(self, config_path, default_config_path=None):
            super(ExampleCustomConfig, self).__init__(config_path, default_config_path)

        def post_process_dinamic_params(self):
            # Breaks down any paths written with '/' and uses path.join to ensure they will work in other operating
            # systems.
            for att, value in self.__dict__.items():
                if isinstance(value, str) and '/' in value:
                    ori = value
                    new_val = path.join(*value.split('/'))
                    if ori != new_val:
                        setattr(self, att, new_val)
                        print(
                            'ATENTION: {} path in config has been reinterpreted as {} to match operating system '
                            'requirements'.format(ori, new_val))
            # General post processing
            if '{}' in self.project_label:
                config_name = path.basename(config_path).replace('config_', '').replace('.yaml', '')
                self.project_label = self.project_label.format(config_name)
            self.project_root = self.default_project_root()
            self.dataset = path.join(self.project_root, self.dataset_folder, self.training_dataset_path)
            self.dataset_folder = absolute(self.dataset_folder)
            self.timeout = self.timeout_debug if self.debug else self.timeout
            self.timeout = None if self.timeout == 0 else self.timeout
            self.project_label = self.date + '_' + self.project_label
            self.results_folder = path.join(absolute(self.results_folder), self.iteration_name + "_last_test") \
                if self.debug else \
                path.join(absolute(self.results_folder), self.project_label)
            self.results_tables_folder = path.join(self.results_folder, self.results_tables_folder)
            self.temporal_ec_path = path.join(self.results_folder, 'execution_complete_temporal.csv')
            self.debug_filepath = path.join(self.results_folder, 'debug.csv')
            self.valid_train_data = path.join(self.results_folder, "valid_train_data")
            self.specifics_folder = absolute(path.join('data', 'specifics', self.country))

        def config_compliance_checks(self):
            assert not (self.technique['name'] == 'by_presentation' and self.technique[
                'target_column'] != 'winning_price'), \
                'If you run a tehcnique by presentation you cannot use a target normalized by mg'


    my_project_configuration = ExampleCustomConfig('path_to_yaml.yaml', 'path_to_default_yaml.yaml')
