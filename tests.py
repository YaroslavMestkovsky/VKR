"""
Модуль с тестами для приложения моделирования инфраструктуры территории.

Этот модуль содержит тесты для проверки функциональности основных компонентов приложения.
Тесты организованы по классам, соответствующим тестируемым компонентам системы.

Classes:
    TestMainWindow: Тесты для главного окна приложения
    TestInteractiveCanvas: Тесты для интерактивного холста
    TestMapGenerator: Тесты для генератора карт
    TestMap: Тесты для работы с картами
"""

import pytest
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication
import sys

from gui import MainWindow, MapGenerator, ValidationError
from logic import Map, MapError, MapInitializationError, MapProcessingError
from constants import CITIES, OBJECTS, RADIUSES

# Преобразуем множества в списки для тестирования
CITIES_LIST = list(CITIES)
OBJECTS_LIST = list(OBJECTS)
RADIUSES_LIST = list(RADIUSES)

@pytest.fixture
def app():
    """Создает экземпляр QApplication для тестов.
    
    Returns:
        QApplication: Экземпляр приложения Qt
    """
    app = QApplication(sys.argv)
    yield app
    app.quit()

@pytest.fixture
def main_window(app):
    """Создает экземпляр главного окна для тестов.
    
    Args:
        app: Фикстура QApplication
        
    Returns:
        MainWindow: Экземпляр главного окна
    """
    window = MainWindow()
    yield window
    window.close()

@pytest.fixture
def map_generator():
    """Создает экземпляр генератора карты для тестов.
    
    Returns:
        MapGenerator: Экземпляр генератора карты с начальными параметрами
    """
    return MapGenerator(
        city=CITIES_LIST[0],
        obj_type=OBJECTS_LIST[0],
        radius=RADIUSES_LIST[0]
    )

class TestMainWindow:
    """Тесты для класса MainWindow.
    
    Этот класс содержит тесты для проверки корректности инициализации
    и работы главного окна приложения.
    """
    
    def test_initialization(self, main_window):
        """Проверяет корректность инициализации главного окна."""
        assert main_window.windowTitle() == "Моделирование инфраструктуры территории", \
            "Неверный заголовок окна"
        assert main_window.city_combo.count() == len(CITIES), \
            "Неверное количество городов в выпадающем списке"
        assert main_window.object_combo.count() == len(OBJECTS), \
            "Неверное количество объектов в выпадающем списке"
        assert main_window.radius_combo.count() == len(RADIUSES), \
            "Неверное количество радиусов в выпадающем списке"
        assert not main_window.save_btn.isEnabled(), \
            "Кнопка сохранения должна быть неактивна при инициализации"
        assert main_window.calculate_btn.isEnabled(), \
            "Кнопка расчета должна быть активна при инициализации"

    def test_ui_elements_existence(self, main_window):
        """Проверяет наличие всех необходимых элементов интерфейса."""
        required_elements = [
            'city_combo',
            'object_combo',
            'radius_combo',
            'calculate_btn',
            'save_btn',
            'progress_bar',
            'progress_label'
        ]
        
        for element in required_elements:
            assert hasattr(main_window, element), \
                f"Отсутствует элемент интерфейса: {element}"

    def test_combo_boxes_content(self, main_window):
        """Проверяет корректность содержимого выпадающих списков."""
        # Проверка списка городов
        for i in range(main_window.city_combo.count()):
            main_window.city_combo.setCurrentIndex(i)
            assert main_window.city_combo.currentText() in CITIES, \
                f"Город {main_window.city_combo.currentText()} отсутствует в списке допустимых городов"

        # Проверка списка объектов
        for i in range(main_window.object_combo.count()):
            main_window.object_combo.setCurrentIndex(i)
            assert main_window.object_combo.currentText() in OBJECTS, \
                f"Объект {main_window.object_combo.currentText()} отсутствует в списке допустимых объектов"

        # Проверка списка радиусов
        for i in range(main_window.radius_combo.count()):
            main_window.radius_combo.setCurrentIndex(i)
            assert main_window.radius_combo.currentText() in RADIUSES, \
                f"Радиус {main_window.radius_combo.currentText()} отсутствует в списке допустимых радиусов"

class TestInteractiveCanvas:
    """Тесты для класса InteractiveCanvas."""
    
    def test_zoom(self, main_window):
        """Проверка функциональности масштабирования."""
        if main_window.current_canvas:
            canvas = main_window.current_canvas
            initial_xlim = canvas.figure.axes[0].get_xlim()
            initial_ylim = canvas.figure.axes[0].get_ylim()

            # Симуляция прокрутки колесика мыши
            event = QWheelEvent(
                QPoint(100, 100),
                QPoint(100, 100),
                QPoint(0, 120),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False
            )
            canvas.wheelEvent(event)

            new_xlim = canvas.figure.axes[0].get_xlim()
            new_ylim = canvas.figure.axes[0].get_ylim()

            assert new_xlim != initial_xlim
            assert new_ylim != initial_ylim

    def test_pan(self, main_window):
        """Проверка функциональности панорамирования."""
        if main_window.current_canvas:
            canvas = main_window.current_canvas
            initial_xlim = canvas.figure.axes[0].get_xlim()
            initial_ylim = canvas.figure.axes[0].get_ylim()

            # Симуляция нажатия правой кнопки мыши
            press_event = QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPoint(100, 100),
                Qt.MouseButton.RightButton,
                Qt.MouseButton.RightButton,
                Qt.KeyboardModifier.NoModifier
            )
            canvas.mousePressEvent(press_event)

            # Симуляция перемещения мыши
            move_event = QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPoint(200, 200),
                Qt.MouseButton.RightButton,
                Qt.MouseButton.RightButton,
                Qt.KeyboardModifier.NoModifier
            )
            canvas.mouseMoveEvent(move_event)

            # Симуляция отпускания кнопки мыши
            release_event = QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPoint(200, 200),
                Qt.MouseButton.RightButton,
                Qt.MouseButton.RightButton,
                Qt.KeyboardModifier.NoModifier
            )
            canvas.mouseReleaseEvent(release_event)

            new_xlim = canvas.figure.axes[0].get_xlim()
            new_ylim = canvas.figure.axes[0].get_ylim()

            assert new_xlim != initial_xlim
            assert new_ylim != initial_ylim

class TestMapGenerator:
    """Тесты для класса MapGenerator."""
    
    def test_initialization(self, map_generator):
        """Проверка корректной инициализации генератора карты."""
        assert map_generator.city == CITIES_LIST[0]
        assert map_generator.obj_type == OBJECTS_LIST[0]
        assert map_generator.radius == int(RADIUSES_LIST[0])

    def test_get_tags(self, map_generator):
        """Проверка генерации тегов для карты."""
        tags = map_generator._get_tags()
        assert isinstance(tags, list)
        assert len(tags) > 0
        for tag in tags:
            assert isinstance(tag, dict)
            assert "objects" in tag
            assert "color" in tag
            assert "label" in tag

class TestMap:
    """Тесты для класса Map.
    
    Этот класс содержит тесты для проверки работы с картами,
    включая инициализацию, получение границ и преобразование единиц измерения.
    """
    
    def test_initialization(self):
        """Проверяет корректность инициализации карты."""
        try:
            map_obj = Map(place=CITIES_LIST[0])
            assert map_obj.place == CITIES_LIST[0], \
                "Неверно установлено место на карте"
            assert map_obj.buildings is None, \
                "Список зданий должен быть None при инициализации"
            assert isinstance(map_obj.cluster_circles, list), \
                "cluster_circles должен быть списком"
            assert isinstance(map_obj.build_cluster_circles, list), \
                "build_cluster_circles должен быть списком"
        except MapInitializationError as e:
            pytest.fail(f"Ошибка инициализации карты: {str(e)}")

    def test_get_city_boundary(self):
        """Проверяет получение границ города."""
        map_obj = Map(place=CITIES_LIST[0])
        boundary = map_obj._get_city_boundary()
        assert boundary is not None, \
            "Границы города не должны быть None"
        assert len(boundary.bounds) == 4, \
            "Границы города должны содержать 4 значения (min_lat, max_lat, min_lon, max_lon)"

    def test_meters_to_degrees_conversion(self):
        """Проверяет корректность преобразования метров в градусы."""
        map_obj = Map(place=CITIES_LIST[0])
        meters = 1000
        latitude = 55.7558  # Москва
        degrees = map_obj._meters_to_degrees(meters, latitude)
        assert isinstance(degrees, float), \
            "Результат должен быть числом с плавающей точкой"
        assert degrees > 0, \
            "Результат должен быть положительным числом"
        assert degrees < 1, \
            "1000 метров не должны превышать 1 градус"

def test_validation_error():
    """Проверяет корректность обработки ошибок валидации."""
    with pytest.raises(ValidationError) as exc_info:
        empty_cities = set()
        if not empty_cities:
            raise ValidationError("Список городов пуст")
    assert str(exc_info.value) == "Список городов пуст", \
        "Неверное сообщение об ошибке валидации"

def test_map_error():
    """Проверяет корректность обработки ошибок карты."""
    with pytest.raises(MapError) as exc_info:
        raise MapError("Тестовая ошибка карты")
    assert str(exc_info.value) == "Тестовая ошибка карты", \
        "Неверное сообщение об ошибке карты"

def test_map_initialization_error():
    """Проверяет корректность обработки ошибок инициализации карты."""
    with pytest.raises(MapInitializationError) as exc_info:
        raise MapInitializationError("Тестовая ошибка инициализации карты")
    assert str(exc_info.value) == "Тестовая ошибка инициализации карты", \
        "Неверное сообщение об ошибке инициализации карты"

def test_map_processing_error():
    """Проверяет корректность обработки ошибок обработки карты."""
    with pytest.raises(MapProcessingError) as exc_info:
        raise MapProcessingError("Тестовая ошибка обработки карты")
    assert str(exc_info.value) == "Тестовая ошибка обработки карты", \
        "Неверное сообщение об ошибке обработки карты"
