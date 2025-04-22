import overpy
import matplotlib.pyplot as plt

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
        """
        Отрисовывает объекты на графике.
        :param result: Результат запроса Overpass API.
        """
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
                ax.plot(lons, lats, 'r-', linewidth=2)  # Красные линии для путей
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
        """
        Основная функция для получения данных и их визуализации.
        :param bbox: tuple(min_lat, min_lon, max_lat, max_lon) - координаты квадрата.
        """
        print("Получение данных из OpenStreetMap...")
        result = self.fetch_data(bbox)
        print(f"Найдено узлов: {len(result.nodes)}, путей: {len(result.ways)}")
        print("Отрисовка объектов...")
        self.plot_objects(result)


# Пример использования
if __name__ == "__main__":
    # Задаем координаты квадрата (широта, долгота)
    bbox = (55.750, 37.600, 55.760, 37.610)  # Центр Москвы

    # Создаем экземпляр класса и запускаем визуализацию
    visualizer = OSMVisualizer()
    visualizer.visualize_bbox(bbox)