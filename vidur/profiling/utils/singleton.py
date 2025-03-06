"""
单例模式实现
"""

class Singleton(type):
    """
    单例元类
    确保一个类只有一个实例，并提供对该实例的全局访问点
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        重写类的实例化方法

        当类被调用时，如果已经存在实例，则返回该实例，否则创建新实例
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
