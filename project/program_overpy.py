import overpy
import matplotlib.pyplot as plt
from math import cos, radians


class OSMVisualizer:
    def __init__(self):
        self.api = overpy.Overpass()

    def fetch_data(self, bbox):
        """
        Получает данные из OpenStreetMap для заданного bounding box.
        :param bbox: tuple(min_lat, min_lon, max_lat, max_lon) - координаты квадрата.
        :return: Результат запроса Overpass API.
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        query = f"""
        [out:json];
        (
            node({min_lat}, {min_lon}, {max_lat}, {max_lon});
            way({min_lat}, {min_lon}, {max_lat}, {max_lon});
        );
        out body;
        """
        return self.api.query(query)

    def plot_objects(self, result):
        """Отрисовывает объекты на графике."""
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(10, 10))

        # Отрисовываем узлы
        for node in result.nodes:
            lat = float(node.lat)
            lon = float(node.lon)
            ax.plot(lon, lat, 'bo', markersize=5)  # Синие точки для узлов

        # Отрисовываем пути
        for way in result.ways:
            try:
                # Получаем координаты всех узлов в пути
                coords = [(float(node.lat), float(node.lon)) for node in way.nodes]
                lats, lons = zip(*coords)  # Разделяем широты и долготы
                ax.plot(lons, lats, 'r-', linewidth=0.5)  # Красные линии для путей
            except Exception as e:
                print(f"Ошибка при обработке пути {way.id}: {e}")

        # Настройка графика
        ax.set_xlabel("Долгота")
        ax.set_ylabel("Широта")
        ax.set_title("Объекты внутри заданного квадрата")
        ax.grid(True)
        plt.axis("equal")  # Сохраняем пропорции между широтой и долготой
        plt.show()

    def visualize_bbox(self, bbox):
        """Основная функция для получения данных и их визуализации."""

        print("Получение данных из OpenStreetMap...")
        result = self.fetch_data(bbox)
        print(f"Найдено узлов: {len(result.nodes)}, путей: {len(result.ways)}")
        print("Отрисовка объектов...")
        self.plot_objects(result)

    @staticmethod
    def calculate_bbox(latitude, longitude, delta=0.01):
        """Рассчитывает bounding box на основе центральных координат и дельты."""

        min_lat = latitude - delta
        max_lat = latitude + delta
        min_lon = longitude - delta / abs(cos(radians(latitude)))
        max_lon = longitude + delta / abs(cos(radians(latitude)))

        return min_lat, min_lon, max_lat, max_lon


# Пример использования
if __name__ == "__main__":
    visualizer = OSMVisualizer()
    bbox = visualizer.calculate_bbox(48.7081906, 44.5153353, delta=0.01) # Центр Волгограда

    visualizer.visualize_bbox(bbox)