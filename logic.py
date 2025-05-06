"""
Модуль для работы с картами и геоданными.

Этот модуль предоставляет функциональность для работы с OpenStreetMap данными,
кластеризации объектов и визуализации результатов на карте.
"""

# Стандартные библиотеки
import logging
from functools import lru_cache, wraps
from time import time
from typing import List, Dict, Any, Optional, Callable, Tuple

# Сторонние библиотеки
import concurrent
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import unary_union
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from osmnx._errors import InsufficientResponseError
from matplotlib.patches import Polygon as MplPolygon

# Локальные импорты
from utils import GeolocatorException, MapError, MapInitializationError, MapProcessingError
from constants import DISPLAY_SETTINGS, CLUSTER_SETTINGS


# Настройки логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def timing(f: Callable) -> Callable:
    """Декоратор для измерения времени выполнения функции.
    
    Args:
        f: Функция, время выполнения которой нужно измерить
        
    Returns:
        Обёрнутая функция с логированием времени выполнения
    """
    @wraps(f)
    def wrap(*args: Any, **kw: Any) -> Any:
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info(f'Функция {f.__name__} выполнилась за {te - ts:.2f} секунд')
        return result
    return wrap

class Map:
    """Класс для работы с картами и геоданными.
    
    Этот класс предоставляет функциональность для:
    - Загрузки границ города
    - Построения графа дорог
    - Кластеризации объектов
    - Визуализации результатов на карте
    
    Attributes:
        buildings (Optional[GeoDataFrame]): Данные о зданиях
        cluster_circles (List[Polygon]): Список кластеров
        build_cluster_circles (List[Polygon]): Список кластеров застройки
        place (str): Название места для анализа
        _cached_area (Optional[Polygon]): Кэшированные границы города
        _cached_road_graph (Optional[Any]): Кэшированный граф дорог
        _cached_colors (Dict[Tuple[str, float], Tuple[float, float, float, float]]): Кэш цветов
        geolocator (Nominatim): Геолокатор для определения координат
    """
    
    def __init__(self, place: str, figsize: Tuple[int, int] = DISPLAY_SETTINGS['FIG_SIZE']) -> None:
        """Инициализация объекта Map.
        
        Args:
            place: Название места для анализа
            figsize: Размер фигуры matplotlib в дюймах
            
        Raises:
            MapInitializationError: При ошибке инициализации карты
        """
        try:
            # Инициализация атрибутов
            self.buildings: Optional[GeoDataFrame] = None
            self.cluster_circles: List[Polygon] = []
            self.build_cluster_circles: List[Polygon] = []
            self.place: str = place
            self._cached_area: Optional[Polygon] = None
            self._cached_road_graph: Optional[Any] = None
            self._cached_colors: Dict[Tuple[str, float], Tuple[float, float, float, float]] = {}
            self.geolocator: Nominatim = Nominatim(user_agent="test_agent")
            
            # Загрузка данных
            self.area: Polygon = self._get_city_boundary()
            plt.switch_backend('Agg')
            self.figure, self.axes = plt.subplots(figsize=figsize)
            self.road_graph = self._get_road_graph()
            if self.road_graph:
                ox.plot_graph(
                    self.road_graph,
                    ax=self.axes,
                    node_size=0,
                    edge_color="gray",
                    show=False
                )
            self.building_sindex = None
        except Exception as e:
            logger.error(f"Ошибка при инициализации карты: {e}")
            raise MapInitializationError(f"Не удалось инициализировать карту: {str(e)}")

    @timing
    def draw(self, _tags: List[Dict[str, Any]],
             progress_callback: Optional[Callable[[int, str], None]] = None) -> None:
        """Отрисовка карты с объектами.

        Args:
            _tags: Список тегов для отображения
            progress_callback: Функция обратного вызова для отображения прогресса

        Raises:
            MapProcessingError: При ошибке отрисовки карты
        """
        try:
            if progress_callback:
                progress_callback(0, "Начало построения карты...")
            self._get_points_parallel(_tags)
            if progress_callback:
                progress_callback(20, "Обработка данных...")

            self.buildings = next((tag["points"] for tag in _tags if tag["label"] == "Здания"), None)
            if self.buildings is not None:
                self.buildings = self.buildings.simplify(tolerance=DISPLAY_SETTINGS['TOLERANCE'])

            build_points = []
            for tag in _tags:
                if tag.get("build", False) and isinstance(tag["points"], GeoDataFrame):
                    build_points.append(tag["points"])

            if progress_callback:
                progress_callback(50, "Отрисовка объектов...")
            self._add_points([_tag for _tag in _tags if isinstance(_tag["points"], GeoDataFrame)])

            if progress_callback:
                progress_callback(70, "Объединение кластеров...")
            self.cluster_circles = self._merge_overlapping_clusters(self.cluster_circles)
            self.build_cluster_circles = self._merge_overlapping_clusters(self.build_cluster_circles)

            if self.buildings is not None and (self.cluster_circles or self.build_cluster_circles):
                self._highlight_buildings()

            plt.title(f"Результаты поиска по запросу {self.place}")
            plt.legend()
            self._set_map_aspect()
            plt.tight_layout()
            if progress_callback:
                progress_callback(100, "Готово!")
        except Exception as e:
            logger.error(f"Ошибка при отрисовке карты: {e}")
            raise MapProcessingError(f"Не удалось отрисовать карту: {str(e)}")

    @lru_cache(maxsize=1)
    def _get_city_boundary(self) -> Polygon:
        """Получение границ города.
        
        Returns:
            Геометрия границ города в виде объекта shapely
            
        Raises:
            GeolocatorException: Если не удалось получить границы города
        """
        if self._cached_area is not None:
            return self._cached_area
        try:
            area = ox.geocode_to_gdf(self.place)
            result = area.geometry.iloc[0]
            self._cached_area = result
            return result
        except Exception as e:
            logger.error(f"Не удалось получить границы города: {e}")
            raise GeolocatorException(f"Ошибка при получении границ города: {str(e)}")

    @lru_cache(maxsize=1)
    def _get_road_graph(self) -> Optional[Any]:
        """Получение графа дорог города.
        
        Returns:
            Граф дорог города или None в случае ошибки
        """
        if self._cached_road_graph is not None:
            return self._cached_road_graph
        try:
            road_graph = ox.graph_from_polygon(self.area, network_type="drive")
            self._cached_road_graph = road_graph
            return road_graph
        except Exception as e:
            logger.error(f"Не удалось получить граф дорог: {e}")
            return None

    def _to_projected_crs(self, gdf: GeoDataFrame) -> GeoDataFrame:
        """Преобразование GeoDataFrame в проекцию UTM.
        
        Args:
            gdf: GeoDataFrame для преобразования
            
        Returns:
            Преобразованный GeoDataFrame в проекции UTM
            
        Raises:
            MapProcessingError: При ошибке преобразования координат
        """
        try:
            if gdf.crs is None:
                return gdf
            center = gdf.geometry.centroid
            lon, lat = center.x.mean(), center.y.mean()
            utm_zone = int((lon + 180) / 6) + 1
            hemisphere = 'north' if lat >= 0 else 'south'
            epsg_code = f'326{utm_zone:02d}' if hemisphere == 'north' else f'327{utm_zone:02d}'
            return gdf.to_crs(epsg=epsg_code)
        except Exception as e:
            logger.error(f"Ошибка при преобразовании координат: {e}")
            raise MapProcessingError(f"Не удалось преобразовать координаты: {str(e)}")

    def _optimize_clusters(self, points: GeoDataFrame, radius_meters: float) -> List[Polygon]:
        """Оптимизация кластеров методом k-means.
        
        Args:
            points: Точки для кластеризации
            radius_meters: Радиус кластера в метрах
            
        Returns:
            Список кластеров в виде кругов
            
        Raises:
            MapProcessingError: При ошибке кластеризации
        """
        try:
            if len(points) == 0:
                return []

            points_projected = self._to_projected_crs(points)
            coords = np.round(np.array(points_projected.geometry.centroid.apply(lambda p: [p.x, p.y]).tolist()), 2)

            best_k, best_score = 1, float('inf')
            max_k = min(DISPLAY_SETTINGS['MAX_CLUSTERS'], len(points))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._evaluate_k, k, coords, radius_meters) for k in range(1, max_k + 1)]
                for future in concurrent.futures.as_completed(futures):
                    k, score = future.result()
                    if score < best_score:
                        best_score, best_k = score, k

            kmeans = MiniBatchKMeans(
                n_clusters=best_k,
                random_state=CLUSTER_SETTINGS['RANDOM_STATE'],
                batch_size=CLUSTER_SETTINGS['BATCH_SIZE']
            )
            kmeans.fit(coords)

            circles = []
            for center in kmeans.cluster_centers_:
                point = Point(center[0], center[1])
                circle = point.buffer(radius_meters)
                circle_gdf = GeoDataFrame(geometry=[circle], crs=points_projected.crs)
                circles.append(circle_gdf.to_crs(points.crs).geometry.iloc[0])

            logger.info(f"Оптимальное количество кластеров: {best_k}")
            return circles
        except Exception as e:
            logger.error(f"Ошибка при кластеризации: {e}")
            raise MapProcessingError(f"Не удалось выполнить кластеризацию: {str(e)}")

    def _evaluate_k(self, k: int, coords: np.ndarray, radius_meters: float) -> Tuple[int, float]:
        """Оценка качества кластеризации для заданного k.
        
        Args:
            k: Количество кластеров
            coords: Координаты точек
            radius_meters: Радиус кластера в метрах
            
        Returns:
            Кортеж (k, score), где score - оценка качества кластеризации
        """
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=CLUSTER_SETTINGS['RANDOM_STATE'],
                batch_size=CLUSTER_SETTINGS['BATCH_SIZE']
            )
            kmeans.fit(coords)
            max_distances = []
            for i in range(k):
                cluster_points = coords[kmeans.labels_ == i]
                if len(cluster_points) > 0:
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    max_distances.append(np.max(distances))
            if max(max_distances) > radius_meters:
                return k, float('inf')
            return k, kmeans.inertia_
        except Exception as e:
            logger.error(f"Ошибка при оценке кластеризации: {e}")
            return k, float('inf')

    def _merge_overlapping_clusters(self, circles: List[Polygon]) -> List[Polygon]:
        """Объединение перекрывающихся кластеров.
        
        Args:
            circles: Список кластеров для объединения
            
        Returns:
            Список объединенных кластеров
            
        Raises:
            MapProcessingError: При ошибке объединения кластеров
        """
        try:
            if not circles:
                return []
            multi_polygon = MultiPolygon(circles)
            merged = unary_union(multi_polygon)
            result = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
            logger.info(f"Объединение кластеров: было {len(circles)}, стало {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Ошибка при объединении кластеров: {e}")
            raise MapProcessingError(f"Не удалось объединить кластеры: {str(e)}")

    def _get_points_parallel(self, tags: List[Dict[str, Any]]) -> None:
        """Параллельное получение точек для каждого тега.
        
        Args:
            tags: Список тегов для поиска объектов
            
        Raises:
            MapProcessingError: При ошибке получения точек
        """
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_tag = {
                    executor.submit(ox.features_from_polygon, self.area, tags=tag["objects"]): tag
                    for tag in tags
                }
                for future in concurrent.futures.as_completed(future_to_tag):
                    tag = future_to_tag[future]
                    try:
                        tag["points"] = future.result()
                    except InsufficientResponseError:
                        tag["points"] = None
                        logger.warning(f"Не найдено ни одного объекта {tag['label']}")
        except Exception as e:
            logger.error(f"Ошибка при получении точек: {e}")
            raise MapProcessingError(f"Не удалось получить точки: {str(e)}")

    def _get_cached_color(self, color_name: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
        """Получение цвета из кэша или создание нового.
        
        Args:
            color_name: Название цвета
            alpha: Прозрачность цвета
            
        Returns:
            Кортеж (r, g, b, a) с компонентами цвета
            
        Raises:
            MapProcessingError: При ошибке получения цвета
        """
        try:
            key = (color_name, alpha)
            if key not in self._cached_colors:
                self._cached_colors[key] = mcolors.to_rgba(color_name, alpha)
            return self._cached_colors[key]
        except Exception as e:
            logger.error(f"Ошибка при получении цвета: {e}")
            raise MapProcessingError(f"Не удалось получить цвет: {str(e)}")

    def _batch_plot(self, gdf: GeoDataFrame, color: str, alpha: float = DISPLAY_SETTINGS['ALPHA'],
                   edgecolor: str = DISPLAY_SETTINGS['EDGE_COLOR'],
                   linewidth: float = DISPLAY_SETTINGS['LINE_WIDTH'],
                   batch_size: int = CLUSTER_SETTINGS['BATCH_SIZE']) -> None:
        """Пакетная отрисовка объектов на карте.
        
        Args:
            gdf: GeoDataFrame с объектами для отрисовки
            color: Цвет объектов
            alpha: Прозрачность
            edgecolor: Цвет границ
            linewidth: Толщина линий
            batch_size: Размер пакета для отрисовки
            
        Raises:
            MapProcessingError: При ошибке отрисовки
        """
        try:
            if gdf.empty:
                return
            for i in range(0, len(gdf), batch_size):
                batch = gdf.iloc[i:i + batch_size]
                batch.plot(
                    ax=self.axes,
                    color=color,
                    alpha=alpha,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )
        except Exception as e:
            logger.error(f"Ошибка при отрисовке объектов: {e}")
            raise MapProcessingError(f"Не удалось отрисовать объекты: {str(e)}")

    def _highlight_buildings(self) -> None:
        """Выделение зданий в зависимости от их расположения относительно кластеров."""
        if self.buildings is None:
            return

        def check_within(gdf: GeoDataFrame, circles: List[Polygon]) -> GeoDataFrame:
            """Проверка, какие здания находятся внутри кругов.
            
            Args:
                gdf: GeoDataFrame с зданиями
                circles: Список кругов-кластеров
                
            Returns:
                GeoDataFrame с зданиями внутри кругов
            """
            mask = np.zeros(len(gdf), dtype=bool)
            for circle in circles:
                mask |= gdf.geometry.apply(lambda x: x.within(circle))
            return gdf[mask]

        buildings_in_clusters = check_within(self.buildings, self.cluster_circles)
        buildings_outside = self.buildings[~self.buildings.index.isin(buildings_in_clusters.index)]
        buildings_in_build = check_within(buildings_outside, self.build_cluster_circles)
        buildings_completely_outside = buildings_outside[~buildings_outside.index.isin(buildings_in_build.index)]

        self._batch_plot(buildings_completely_outside, "gray", 0.7)
        self._batch_plot(buildings_in_build, "lime", 0.7)

        self.axes.plot([], [], color="gray", alpha=0.7,
                      label=f"Здания вне радиусов ({len(buildings_completely_outside)})")
        self.axes.plot([], [], color="lime", alpha=0.7,
                      label=f"Здания в радиусах застройки ({len(buildings_in_build)})")

    def _set_map_aspect(self) -> None:
        """Установка правильного соотношения сторон карты."""
        try:
            x_min, x_max = self.axes.get_xlim()
            y_min, y_max = self.axes.get_ylim()
            if all(np.isfinite([x_min, x_max, y_min, y_max])):
                width = x_max - x_min
                height = y_max - y_min
                aspect = width / height
                self.axes.set_aspect(max(0.1, min(10.0, aspect)))
            else:
                self.axes.set_aspect('auto')
        except Exception as e:
            self.axes.set_aspect('auto')
            logger.error(f"Ошибка при установке аспекта: {e}")

    def _add_points(self, data: List[Dict[str, Any]]) -> None:
        """Добавление точек на карту.
        
        Args:
            data: Список словарей с данными для отображения
        """
        all_clusters = []
        for info in data:
            if info["is_cluster"]:
                for idx, obj in info["points"].iterrows():
                    center = obj.geometry.centroid
                    radius_degrees = self._meters_to_degrees(info["cluster_size"], center.y)
                    circle = center.buffer(radius_degrees)
                    all_clusters.append((circle, info["color"], info.get("build", False)))

        if all_clusters:
            build_circles = [c for c, _, b in all_clusters if b]
            regular_circles = [c for c, _, b in all_clusters if not b]

            self.build_cluster_circles = self._merge_overlapping_clusters(build_circles)
            self.cluster_circles = self._merge_overlapping_clusters(regular_circles)

            # Отрисовка обычных кластеров
            cluster_patches = [
                MplPolygon(np.array(circle.exterior.coords), closed=True, fill=False)
                for circle in self.cluster_circles
            ]

            # Отрисовка кластеров застройки с зданиями
            build_patches = []
            for num, circle in enumerate(self.build_cluster_circles, start=1):
                # Найти места для зданий
                building_locations = self._find_building_locations(
                    circle,
                    self.buildings,
                    grid_size=DISPLAY_SETTINGS['GRID_SIZE'],
                    min_buildings=DISPLAY_SETTINGS['MIN_BUILDINGS'],
                    max_buildings=DISPLAY_SETTINGS['MAX_BUILDINGS']
                )

                # Отрисовка зданий
                for i, (point, scores) in enumerate(building_locations):
                    building_size = DISPLAY_SETTINGS['BUILDING_SIZE']  # Размер здания в градусах (~50 метров)
                    building = point.buffer(building_size, cap_style=3)

                    # Преобразуем в GeoDataFrame для отрисовки
                    building_gdf = GeoDataFrame(geometry=[building], crs=self.buildings.crs)
                    building_gdf.plot(
                        ax=self.axes,
                        color="red",
                        alpha=0.7,
                        edgecolor="black",
                        linewidth=1,
                        zorder=3
                    )

                    # Добавляем аннотацию
                    location_rating = self._get_location_rating(scores)
                    self.axes.annotate(
                        text=f"Кластер №{num}\nВариант {i + 1}\n{location_rating}",
                        xy=(point.x, point.y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        fontsize=8,
                        color="black",
                        ha="center",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
                    )

                # Добавляем кластер в патчи
                build_patches.append(MplPolygon(np.array(circle.exterior.coords), closed=True, fill=False))

            # Отрисовка коллекций
            if cluster_patches:
                cluster_collection = PatchCollection(
                    cluster_patches,
                    facecolor='none',
                    edgecolor='red',
                    linewidth=0.5,
                    alpha=0.8,
                    hatch='\\\\'
                )
                self.axes.add_collection(cluster_collection)

            if build_patches:
                build_collection = PatchCollection(
                    build_patches,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.4,
                    hatch='/'
                )
                self.axes.add_collection(build_collection)

        # Отрисовка остальных объектов
        for info in data:
            if info["points"] is not None and not info["points"].empty:
                self._batch_plot(
                    info["points"],
                    info["color"],
                    info.get('alpha', DISPLAY_SETTINGS['ALPHA']),
                    DISPLAY_SETTINGS['EDGE_COLOR'],
                    DISPLAY_SETTINGS['LINE_WIDTH']
                )
                if info["need_label"]:
                    for _, obj in info["points"].iterrows():
                        name = obj.get("name", "null")
                        x, y = obj.geometry.centroid.x, obj.geometry.centroid.y
                        self.axes.annotate(
                            text=name,
                            xy=(x, y),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=8,
                            color="black",
                            ha="center",
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
                        )

    @staticmethod
    @lru_cache(maxsize=1000)
    def _meters_to_degrees(meters: float, latitude: float) -> float:
        """Преобразование метров в градусы на заданной широте.
        
        Args:
            meters: Расстояние в метрах
            latitude: Широта в градусах
            
        Returns:
            Расстояние в градусах
        """
        return meters / (111319.9 * abs(np.cos(np.radians(latitude))))

    def _find_building_locations(self, circle: Polygon, buildings: GeoDataFrame, 
                               grid_size: int = DISPLAY_SETTINGS['GRID_SIZE'],
                               min_buildings: int = DISPLAY_SETTINGS['MIN_BUILDINGS'],
                               max_buildings: int = DISPLAY_SETTINGS['MAX_BUILDINGS']) -> List[Tuple[Point, Dict[str, float]]]:
        """Поиск оптимальных мест для размещения зданий в кластере.
        
        Args:
            circle: Круг-кластер
            buildings: GeoDataFrame с существующими зданиями
            grid_size: Размер сетки для поиска
            min_buildings: Минимальное количество зданий
            max_buildings: Максимальное количество зданий
            
        Returns:
            Список кортежей (точка, оценки) для размещения зданий
        """
        if buildings.empty:
            return [(circle.centroid, {"proximity": 0, "distance": 0, "total": 0})]

        minx, miny, maxx, maxy = circle.bounds
        x_step = (maxx - minx) / grid_size
        y_step = (maxy - miny) / grid_size

        buildings_in_cluster = buildings[buildings.geometry.apply(lambda x: x.within(circle))]
        if buildings_in_cluster.empty:
            return [(circle.centroid, {"proximity": 0, "distance": 0, "total": 0})]

        buildings_projected = self._to_projected_crs(buildings_in_cluster)
        buildings_center = buildings_projected.geometry.centroid.unary_union.centroid
        buildings_center = GeoDataFrame(geometry=[buildings_center], crs=buildings_projected.crs
                                      ).to_crs(buildings.crs).geometry.iloc[0]

        num_buildings = max(min_buildings, min(max_buildings, len(buildings_in_cluster) // 10))
        building_locations = []
        placed_buffers = []
        buffer_size = min(x_step, y_step) * DISPLAY_SETTINGS['BUFFER_SIZE']

        all_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = minx + (i + 0.5) * x_step
                y = miny + (j + 0.5) * y_step
                point = Point(x, y)
                if not point.within(circle):
                    continue
                buffer = point.buffer(buffer_size)
                all_points.append((point, buffer))

        # Параллельная обработка
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._evaluate_point, all_points, [buildings] * len(all_points),
                                      [buildings_center] * len(all_points), [buffer_size] * len(all_points)))

        valid_points = [res for res in results if res is not None]
        valid_points.sort(key=lambda x: x[1]["total"])

        for point, scores, buffer in valid_points:
            if len(building_locations) >= num_buildings:
                break
            if not any(buffer.intersects(placed) for placed in placed_buffers):
                building_locations.append((point, scores))
                placed_buffers.append(buffer)

        return building_locations

    def _evaluate_point(self, point_buffer: Tuple[Point, Any], buildings: GeoDataFrame,
                       center: Point, buffer_size: float) -> Optional[Tuple[Point, Dict[str, float], Any]]:
        """Оценка точки для размещения здания.
        
        Args:
            point_buffer: Кортеж (точка, буфер)
            buildings: GeoDataFrame с существующими зданиями
            center: Центр кластера
            buffer_size: Размер буфера
            
        Returns:
            Кортеж (точка, оценки, буфер) или None, если точка не подходит
        """
        point, buffer = point_buffer
        buildings_in_buffer = buildings[buildings.geometry.intersects(buffer)]
        proximity = len(buildings_in_buffer)
        distance = point.distance(center) / buffer_size
        total = proximity * 2 + distance
        return (point, {"proximity": proximity, "distance": distance, "total": total}, buffer)

    def _get_location_rating(self, scores: Dict[str, float]) -> str:
        """Получение текстовой оценки местоположения.
        
        Args:
            scores: Словарь с оценками местоположения
            
        Returns:
            Текстовая оценка местоположения
        """
        proximity = scores["proximity"]
        distance = scores["distance"]

        proximity_text = (
            "свободная территория" if proximity == 0 else
            "мало зданий рядом" if proximity == 1 else
            "умеренная застройка" if proximity <= 3 else
            "плотная застройка"
        )

        distance_text = (
            "в центре" if distance < 0.2 else
            "близко к центру" if distance < 0.4 else
            "на среднем расстоянии" if distance < 0.6 else
            "на окраине"
        )

        return f"{proximity_text}, {distance_text}"