"""
Графический интерфейс для моделирования инфраструктуры территории.

Этот модуль предоставляет GUI для визуализации и анализа инфраструктуры
городов с использованием PyQt6 и Matplotlib.
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Tuple

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCursor, QWheelEvent, QMouseEvent, QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from constants import CITIES, OBJECTS, RADIUSES, DISPLAY_SETTINGS, SAVE_SETTINGS, DEFAULT_TAGS, TAGS_MAP
from logic import Map
from utils import ValidationError

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveCanvas(FigureCanvas):
    """Интерактивный холст для отображения и взаимодействия с картой.
    
    Attributes:
        _pan_start (Optional[Tuple[float, float]]): Начальная точка панорамирования.
        _zoom_factor (float): Коэффициент масштабирования.
        _dragging (bool): Флаг перетаскивания.
    """
    
    def __init__(self, figure: Figure, parent: Optional[QWidget] = None) -> None:
        """Инициализация интерактивного холста.
        
        Args:
            figure: Объект фигуры matplotlib.
            parent: Родительский виджет.
            
        Raises:
            ValueError: Если figure не является экземпляром Figure.
        """
        if not isinstance(figure, Figure):
            raise ValueError("figure должен быть экземпляром matplotlib.figure.Figure")
            
        super().__init__(figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._pan_start: Optional[Tuple[float, float]] = None
        self._zoom_factor: float = DISPLAY_SETTINGS['ZOOM_FACTOR']
        self._dragging: bool = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Обработка события колесика мыши для масштабирования.
        
        Args:
            event: Событие колесика мыши.
        """
        try:
            ax = self.figure.axes[0]

            cursor_data = ax.transData.inverted().transform([event.position().x(), event.position().y()])
            x, y = cursor_data[0], cursor_data[1]

            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            if event.angleDelta().y() > 0:
                new_x_min = x - (x - x_min) / self._zoom_factor
                new_x_max = x + (x_max - x) / self._zoom_factor
                new_y_min = y - (y - y_min) / self._zoom_factor
                new_y_max = y + (y_max - y) / self._zoom_factor
            else:
                new_x_min = x - (x - x_min) * self._zoom_factor
                new_x_max = x + (x_max - x) * self._zoom_factor
                new_y_min = y - (y - y_min) * self._zoom_factor
                new_y_max = y + (y_max - y) * self._zoom_factor

            ax.set_xlim([new_x_min, new_x_max])
            ax.set_ylim([new_y_min, new_y_max])
            self.draw()
        except Exception as e:
            logger.error(f"Ошибка при масштабировании: {e}")
            QMessageBox.warning(self, "Ошибка", "Не удалось выполнить масштабирование")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Обработка события нажатия кнопки мыши.
        
        Args:
            event: Событие мыши.
        """
        try:
            if event.button() == Qt.MouseButton.RightButton:
                self._pan_start = (event.position().x(), event.position().y())
                self._dragging = True
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

            super().mousePressEvent(event)
        except Exception as e:
            logger.error(f"Ошибка при обработке нажатия мыши: {e}")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Обработка события перемещения мыши.
        
        Args:
            event: Событие мыши.
        """
        try:
            if self._dragging and self._pan_start:
                ax = self.figure.axes[0]
                x_start, y_start = self._pan_start
                x_end, y_end = event.position().x(), event.position().y()

                dx = ax.transData.inverted().transform([x_start, 0])[0] - ax.transData.inverted().transform([x_end, 0])[0]
                dy = ax.transData.inverted().transform([0, y_end])[1] - ax.transData.inverted().transform([0, y_start])[1]

                ax.set_xlim(ax.get_xlim() + dx)
                ax.set_ylim(ax.get_ylim() + dy)

                self._pan_start = (x_end, y_end)
                self.draw()

            super().mouseMoveEvent(event)
        except Exception as e:
            logger.error(f"Ошибка при перемещении мыши: {e}")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Обработка события отпускания кнопки мыши.
        
        Args:
            event: Событие мыши.
        """
        try:
            if event.button() == Qt.MouseButton.RightButton:
                self._pan_start = None
                self._dragging = False
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

            super().mouseReleaseEvent(event)
        except Exception as e:
            logger.error(f"Ошибка при отпускании кнопки мыши: {e}")


class MapGenerator(QThread):
    """Поток для генерации карты.
    
    Attributes:
        finished (pyqtSignal): Сигнал завершения генерации.
        error (pyqtSignal): Сигнал ошибки.
        progress (pyqtSignal): Сигнал прогресса.
    """
    
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(
        self,
        city: str,
        obj_type: str,
        radius: str,
        parent: Optional[QWidget] = None
    ) -> None:
        """Инициализация генератора карты.
        
        Args:
            city: Название города.
            obj_type: Тип объектов для отображения.
            radius: Радиус поиска в метрах.
            parent: Родительский виджет.
        """
        super().__init__(parent)
        self.city = city
        self.obj_type = obj_type
        self.radius = int(radius)

    def _get_tags(self):
        tags = TAGS_MAP.get(self.obj_type, [])
        tags.extend(DEFAULT_TAGS)
        
        # Установка радиуса для тегов, которые его используют
        for tag in tags:
            if tag.get("is_cluster"):
                tag["cluster_size"] = self.radius
                
        return tags

    def run(self) -> None:
        """Запуск генерации карты в отдельном потоке."""
        try:
            self.progress.emit(0, "Инициализация карты...")
            tags = self._get_tags()

            self.progress.emit(10, "Загрузка границ города...")
            city_map = Map(place=self.city, figsize=DISPLAY_SETTINGS['FIGURE_SIZE'])
            
            self.progress.emit(20, "Построение карты...")
            city_map.draw(tags, progress_callback=lambda p, msg: self.progress.emit(p, msg))

            self.progress.emit(90, "Финальная настройка отображения...")
            city_map.axes.set_position([0, 0, 1, 1])
            city_map.axes.set_aspect('auto')

            self.progress.emit(100, "Готово!")
            self.finished.emit(city_map)
        except Exception as e:
            logger.error(f"Ошибка при генерации карты: {e}")
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    """Главное окно приложения.
    
    Attributes:
        is_maximized (bool): Флаг развернутого окна.
        current_canvas (Optional[InteractiveCanvas]): Текущий холст.
        map_generator (Optional[MapGenerator]): Генератор карты.
        toolbar (Optional[NavigationToolbar2QT]): Панель инструментов.
    """
    
    def __init__(self) -> None:
        """Инициализация главного окна."""
        super().__init__()
        
        self.is_maximized: bool = False
        self.current_canvas: Optional[InteractiveCanvas] = None
        self.map_generator: Optional[MapGenerator] = None
        self.toolbar: Optional[NavigationToolbar2QT] = None
        
        self._setup_ui()
        self._validate_initial_state()

    def _validate_initial_state(self) -> None:
        """Проверка начального состояния приложения.
        
        Raises:
            ValidationError: Если начальное состояние некорректно.
        """
        if not CITIES:
            raise ValidationError("Список городов пуст")
        if not OBJECTS:
            raise ValidationError("Список объектов пуст")
        if not RADIUSES:
            raise ValidationError("Список радиусов пуст")

    def _setup_ui(self) -> None:
        """Настройка пользовательского интерфейса."""
        try:
            self.setWindowTitle("Моделирование инфраструктуры территории")
            self.setWindowFlags(
                Qt.WindowType.WindowMinMaxButtonsHint |
                Qt.WindowType.WindowCloseButtonHint |
                Qt.WindowType.Window
            )
            self.setMinimumSize(*DISPLAY_SETTINGS['MIN_WINDOW_SIZE'])

            central_widget = QWidget()
            self.setCentralWidget(central_widget)

            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)

            tools_panel = self._create_tools_panel()
            main_layout.addWidget(tools_panel)

            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            self.progress_label = QLabel()
            self.progress_label.setVisible(False)
            self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            progress_layout = QVBoxLayout()
            progress_layout.addWidget(self.progress_label)
            progress_layout.addWidget(self.progress_bar)
            
            progress_widget = QWidget()
            progress_widget.setLayout(progress_layout)
            main_layout.addWidget(progress_widget)

            self.plot_container = QWidget()
            self.plot_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            self.plot_layout = QVBoxLayout(self.plot_container)
            self.plot_layout.setContentsMargins(0, 0, 0, 0)
            self.plot_layout.setSpacing(0)

            main_layout.addWidget(self.plot_container, stretch=1)
        except Exception as e:
            logger.error(f"Ошибка при настройке UI: {e}")
            raise

    def _create_tools_panel(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        city_label = QLabel("Город:")
        self.city_combo = QComboBox()
        self.city_combo.addItems(CITIES)

        object_label = QLabel("Объекты:")
        self.object_combo = QComboBox()
        self.object_combo.addItems(OBJECTS)

        radius_label = QLabel("Радиус (м):")
        self.radius_combo = QComboBox()
        self.radius_combo.addItems(RADIUSES)
        self.radius_combo.setCurrentText("500")

        self.calculate_btn = QPushButton("Рассчитать")
        self.calculate_btn.clicked.connect(self.generate_map)

        self.save_btn = QPushButton("Загрузить")
        self.save_btn.clicked.connect(self.save_map)
        self.save_btn.setEnabled(False)

        layout.addWidget(city_label)
        layout.addWidget(self.city_combo)
        layout.addWidget(object_label)
        layout.addWidget(self.object_combo)
        layout.addWidget(radius_label)
        layout.addWidget(self.radius_combo)
        layout.addWidget(self.calculate_btn)
        layout.addWidget(self.save_btn)
        layout.addStretch()

        return panel

    def save_map(self) -> None:
        """Сохраняет текущую карту в файл."""
        logger.info("Попытка сохранения карты")
        
        if not self.current_canvas or not self.current_canvas.figure:
            logger.error("Попытка сохранения без активной карты")
            QMessageBox.warning(self, "Предупреждение", "Нет карты для сохранения")
            return

        try:
            default_name = f"map_{self.city_combo.currentText()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{SAVE_SETTINGS['DEFAULT_EXTENSION']}"
            
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить карту",
                default_name,
                SAVE_SETTINGS['FILE_FILTER']
            )
            
            if file_name:
                self.current_canvas.figure.savefig(
                    file_name,
                    dpi=DISPLAY_SETTINGS['DPI'],
                    bbox_inches='tight',
                    pad_inches=DISPLAY_SETTINGS['PAD_INCHES']
                )
                logger.info(f"Карта успешно сохранена в файл: {file_name}")
                QMessageBox.information(self, "Успех", "Карта успешно сохранена")
        except Exception as e:
            logger.error(f"Ошибка при сохранении карты: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить карту: {str(e)}")

    def generate_map(self) -> None:
        """Запускает процесс генерации карты."""
        self.calculate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.calculate_btn.setText("Построение карты...")
        self.clear_plot_area()

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Подготовка к построению карты...")

        city = self.city_combo.currentText()
        obj_type = self.object_combo.currentText()
        radius = self.radius_combo.currentText()

        self.map_generator = MapGenerator(city, obj_type, radius)
        self.map_generator.finished.connect(self.on_map_generated)
        self.map_generator.error.connect(self.on_map_error)
        self.map_generator.progress.connect(self.update_progress)
        self.map_generator.start()

    @pyqtSlot(int, str)
    def update_progress(self, value: int, message: str) -> None:
        """Обновляет прогресс-бар и сообщение о прогрессе.
        
        Args:
            value: Значение прогресса (0-100).
            message: Сообщение о текущем этапе.
        """
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    @pyqtSlot(object)
    def on_map_generated(self, city_map: Map) -> None:
        """Обработчик завершения генерации карты.
        
        Args:
            city_map: Сгенерированная карта.
        """
        try:
            canvas = InteractiveCanvas(city_map.axes.figure, self.plot_container)
            self.current_canvas = canvas

            self.toolbar = NavigationToolbar2QT(canvas, self)

            self.plot_layout.addWidget(self.toolbar)
            self.plot_layout.addWidget(canvas, stretch=1)

            canvas.draw()
            
            self.save_btn.setEnabled(True)
            
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            
            logger.info("Карта успешно сгенерирована и отображена")
        except Exception as e:
            logger.error(f"Ошибка отображения карты: {e}")
            self.current_canvas = None
            self.save_btn.setEnabled(False)
        finally:
            self.calculate_btn.setEnabled(True)
            self.calculate_btn.setText("Построить карту")

    @pyqtSlot(str)
    def on_map_error(self, error_msg: str) -> None:
        """Обработчик ошибки генерации карты.
        
        Args:
            error_msg: Сообщение об ошибке.
        """
        logger.error(f"Ошибка генерации карты: {error_msg}")
        self.calculate_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.calculate_btn.setText("Построить карту")
        
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        QMessageBox.critical(self, "Ошибка", f"Не удалось построить карту: {error_msg}")

    def clear_plot_area(self) -> None:
        """Очищает область отображения карты."""
        logger.info("Очистка области отображения карты")
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        self.current_canvas = None
        self.toolbar = None
        self.save_btn.setEnabled(False)
        logger.info("Область отображения карты очищена")

    def toggle_maximize(self) -> None:
        """Переключает состояние развертывания окна."""
        if self.is_maximized:
            self.showNormal()
        else:
            self.showMaximized()

        self.is_maximized = not self.is_maximized

    def closeEvent(self, event: QCloseEvent) -> None:
        """Обработчик закрытия окна.
        
        Args:
            event: Событие закрытия.
        """
        if self.map_generator and self.map_generator.isRunning():
            self.map_generator.quit()
            self.map_generator.wait()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
