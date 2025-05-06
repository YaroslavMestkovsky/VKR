class GeolocatorException(Exception):
    """Исключения геолокатора."""
    pass

class MapError(Exception):
    """Базовый класс для исключений модуля Map."""
    pass

class MapInitializationError(MapError):
    """Исключение при инициализации карты."""
    pass

class MapProcessingError(MapError):
    """Исключение при обработке данных карты."""
    pass

class ValidationError(Exception):
    """Исключение для ошибок валидации."""
    pass
