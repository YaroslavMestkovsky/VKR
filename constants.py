"""
Константы для работы с картографическими данными и настройками отображения.

Этот модуль содержит константы для:
- Списка городов и объектов для анализа
- Настроек отображения карт
- Настроек сохранения файлов
- Тегов и стилей для различных объектов на карте
"""

# Список городов для анализа
CITIES = {
    "Аксай",
    "Волгоград",
    "Новгород",
    "Москва",
    "Калининград",
    "Самара",
}

# Список объектов для анализа
OBJECTS = {
    "Школы",
    "Больницы",
    "Поликлиники",
    "Магазины",
    "Светофоры",
    "Школы",
}

# Доступные радиусы поиска (в метрах)
RADIUSES = [
    "100", "200", "300", "400", "500",
    "750", "1000", "1500", "2000"
]

# Настройки сохранения файлов
SAVE_SETTINGS = {
    'FILE_FILTER': "PNG файлы (*.png);;Все файлы (*.*)",
    'DEFAULT_EXTENSION': '.png',
}

# Настройки отображения
DISPLAY_SETTINGS = {
    'FIG_SIZE': (25, 25),
    'BATCH_SIZE': 1000,
    'ALPHA': 0.5,
    'LINE_WIDTH': 0.2,
    'EDGE_COLOR': "black",
    'MAX_CLUSTERS': 10,
    'TOLERANCE': 0.0001,
    'BUFFER_SIZE': 0.1,
    'GRID_SIZE': 10,
    'MIN_BUILDINGS': 1,
    'MAX_BUILDINGS': 3,
    'BUILDING_SIZE': 0.0004,  # Размер здания в градусах (~40 метров)
    'ZOOM_FACTOR': 1.5,
    'FIGURE_SIZE': (5, 5),
    'DPI': 300,
    'PAD_INCHES': 0.1,
    'MIN_WINDOW_SIZE': (800, 600),
}

# Настройки кластеризации
CLUSTER_SETTINGS = {
    'RANDOM_STATE': 42,
    'BATCH_SIZE': 1000,
}

# Базовые теги для отображения на карте
DEFAULT_TAGS = [
    {
        "objects": {"building": True},
        "color": "blue",
        "alpha": 0.5,
        "label": "Здания",
        "need_label": False,
        "is_cluster": False,
        "cluster_size": None,
        "build": False,
    },
    {
        "objects": {"natural": "water"},
        "color": "lightblue",
        "alpha": None,
        "label": "Вода",
        "need_label": False,
        "is_cluster": False,
        "cluster_size": None,
        "build": False,
    },
    {
        "objects": {"natural": "wood"},
        "color": "green",
        "alpha": None,
        "label": "Лес",
        "need_label": False,
        "is_cluster": False,
        "cluster_size": None,
        "build": False,
    },
    {
        "objects": {"landuse": "brownfield"},
        "color": "black",
        "alpha": 0.5,
        "label": "Заброшенные территории",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": True,
    },
    {
        "objects": {"landuse": "greenfield"},
        "color": "black",
        "alpha": 0.5,
        "label": "Незастроенная территория",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": True,
    },
    {
        "objects": {"landuse": "construction"},
        "color": "orange",
        "alpha": 0.5,
        "label": "Строительные площадки",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": True,
    },
    {
        "objects": {"landuse": "vacant"},
        "color": "gray",
        "alpha": 0.5,
        "label": "Пустующие территории",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": True,
    },
]

# Словарь тегов для различных типов объектов
TAGS_MAP = {
    "Парковки": [{
        "objects": {"amenity": "parking"},
        "color": "red",
        "alpha": 0.3,
        "label": "Парковки",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": False,
    }],
    "Аптеки": [{
        "objects": {"amenity": "pharmacy"},
        "color": "yellow",
        "alpha": None,
        "label": "Аптеки",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": False,
    }],
    "Больницы": [{
        "objects": {"amenity": "hospital"},
        "color": "red",
        "alpha": None,
        "label": "Больницы",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": False,
    }],
    "Поликлиники": [{
        "objects": {"amenity": "clinic"},
        "color": "pink",
        "alpha": None,
        "label": "Поликлиники",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": False,
    }],
    "Школы": [{
        "objects": {"amenity": "school"},
        "color": "purple",
        "alpha": None,
        "label": "Школы",
        "need_label": False,
        "is_cluster": True,
        "cluster_size": None,
        "build": False,
    }],
}
