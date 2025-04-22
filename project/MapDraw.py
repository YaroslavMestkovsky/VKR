import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import logging

from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from osmnx._errors import InsufficientResponseError
from project.utils import GeolocatorException


class Map:
    """Класс, рисующий карту по заданным параметрам."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
    )

    def __init__(self, place: str, figsize: tuple = (25, 25), radius: int = 500):
        self.place = place
        self.radius = radius

        self.geolocator = Nominatim(user_agent="test_agent")
        self.center_point = self._get_center_point()

        _, self.axes = plt.subplots(figsize=figsize)

        # Подготовка и отрисовка карты дорог.
        road_graph = ox.graph_from_point(self.center_point, dist=radius, network_type="drive")
        ox.plot_graph(road_graph, ax=self.axes, node_size=0, edge_color="gray", show=False)

    def _get_center_point(self):
        locations = self.geolocator.geocode(self.place, exactly_one=False)

        if locations:
            locations_amount = len(locations)

            if locations_amount == 1:
                latitude, longitude = locations[0].latitude, locations[0].longitude
            else:
                logging.error('Найдено более одного объекта, уточните запрос.')
                raise GeolocatorException
        else:
            logging.error('Не найдено ни одного объекта.')
            raise GeolocatorException

        return latitude, longitude

    def draw(self, _tags: list[dict]):
        self._get_points(_tags)
        self._add_points([_tag for _tag in _tags if isinstance(_tag['objects'], GeoDataFrame)])

        plt.title(f"Результаты поиска по запросу {self.place}")
        plt.legend()
        plt.show()

    def _get_points(self, _tags: list[dict]) -> None:
        """Находим объекты по тегам в заданном радиусе."""

        for _tag in _tags:
            try:
                _tag['objects'] = ox.features_from_point(
                    self.center_point,
                    tags=_tag['objects'],
                    dist=self.radius,
                )
            except InsufficientResponseError:
                logging.error(f'Не найдено ни одного объекта {_tag["label"]}')

    def _add_points(self, data: list[dict]):
        """Размещает точки на карте."""

        for info in data:
            info['objects'].plot(
                ax=self.axes,
                label=info['label'],
                color=info['color'],
                edgecolor='black',
                linewidth=0.2,
            )
            
            if info['need_label']:
                for idx, obj in info['objects'].iterrows():
                    if "name" in obj and not pd.isna(obj["name"]):
                        name = obj["name"]
                    else:
                        name = 'null'

                    x, y = obj.geometry.centroid.x, obj.geometry.centroid.y
                    self.axes.annotate(
                        text=name,
                        xy=(x, y),
                        xytext=(3, 3),  # Смещение текста относительно точки
                        textcoords="offset points",
                        fontsize=10,
                        color="black",
                        ha="right",  # Выравнивание по горизонтали
                        va="bottom"  # Выравнивание по вертикали
                    )

if __name__ == '__main__':
    tags = [
        {
            'objects': {"building": True},
            'color': 'lightblue',
            'label': 'Жилые дома',
            'need_label': False
        },
        {
            'objects': {"amenity": "pharmacy"},
            'color': 'white',
            'label': 'Аптеки',
            'need_label': False
        },
        {
            'objects': {"amenity": "school"},
            'color': 'brown',
            'label': 'Школы',
            'need_label': False
        },
        {
            'objects': {"amenity": "university"},
            'color': 'purple',
            'label': 'Вузы',
            'need_label': True
        },
        {
            'objects': {"shop": ["supermarket", "convenience"]},
            'color': 'green',
            'label': 'Магазины',
            'need_label': True
        },
        {
            'objects': {"highway": "traffic_signals"},
            'color': 'yellow',
            'label': 'Светофоры',
            'need_label': False
        },
    ]

    try:
        # place = 'переулок Ногина, пос. Линейный, Тракторозаводский район, Волгоград, городской округ Волгоград, Волгоградская область, 400088, Россия'
        place = 'Волгоград'
        vlg_map = Map(place=place, radius=400)
        vlg_map.draw(tags)

    except GeolocatorException:
        logging.info('Завершение программы в следствии критической ошибки.')
