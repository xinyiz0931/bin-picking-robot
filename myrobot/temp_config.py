import typing
class Config(object):
    #############################
    # Begin Configuration Section
    #############################

    _FOO: typing.Optional[str] = None
    _BAR: typing.Optional[str] = None

    @classmethod
    def get_foo_var(cls) -> str:
        """Example variable that is set in the config file (preferred)"""
        if cls._FOO is None:
            cls._FOO = Config.get_required_config_var('foo')
        return cls._FOO

    @classmethod
    def get_bar_var(cls) -> str:
        """Example variable that is set via env var (not preferred)"""
        if cls._BAR is None:
            cls._BAR = Config.get_required_env_var('BAR')
        return cls._BAR

    @classmethod
    def get_wuz(cls) -> str:
        if cls._WUZ is None:
            if 'wuz' not in cls._CONFIG:
                cls._WUZ = Config.get_required_env_var('WUZ')
            else:
                cls._WUZ = cls._CONFIG['wuz']
        if not os.path.isdir(cls._WUZ):
            raise Exception(f"Error: Path {cls._WUZ} is not a directory")
        return cls._WUZ
