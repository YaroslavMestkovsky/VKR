import overpy


class Overpy:
    def __init__(self):
        self.api = overpy.Overpass()

    def get_query_result(self, query):
        return self.api.query(query)

    @staticmethod
    def get_query(area, filters:list):
        """Query для запроса в api."""

        if isinstance(filters, list):
            filters = "\n".join(filters)

        return f"""
            [out:json];
            area["name"="{area}"]->.searchArea;
            (
                {filters}
            );
            out body;
        """

    @staticmethod
    def amenity(amenity_type):
        """Тег для объектов с ключом amenity (например, "restaurant", "cafe")."""
        return f'node["amenity"="{amenity_type}"](area.searchArea);'

    @staticmethod
    def highway(highway_type):
        """Тег для объектов с ключом highway (например, "primary", "residential")."""
        return f'way["highway"="{highway_type}"](area.searchArea);'

    @staticmethod
    def route(route_type):
        """Тег для маршрутов с ключом route (например, "bus", "train")."""
        return f'relation["route"="{route_type}"](area.searchArea);'

    @staticmethod
    def leisure(leisure_type):
        """Тег для объектов с ключом leisure (например, "park", "stadium")."""
        return f'node["leisure"="{leisure_type}"](area.searchArea);'

    @staticmethod
    def shop(shop_type):
        """Тег для магазинов с ключом shop (например, "supermarket", "clothes")."""
        return f'node["shop"="{shop_type}"](area.searchArea);'

    @staticmethod
    def natural(natural_type):
        """Тег для природных объектов с ключом natural (например, "wood", "water")."""
        return f'node["natural"="{natural_type}"](area.searchArea);'

    @staticmethod
    def tourism(tourism_type):
        """Тег для туристических объектов с ключом tourism (например, "hotel", "museum")."""
        return f'node["tourism"="{tourism_type}"](area.searchArea);'

    @staticmethod
    def node_by_name(name, exact=True):
        """Тег для объекта с конкретным названием (например, Магнит)."""

        key = "="

        if not exact:
            key = "~"

        return f'node["name"{key}"{name}", i](area.searchArea);'

    @staticmethod
    def bbox_filter(key, value, bbox):
        """Тег для объектов в заданной области."""
        min_lat, min_lon, max_lat, max_lon = bbox
        return f'node["{key}"="{value}"]({min_lat}, {min_lon}, {max_lat}, {max_lon});'


manager = Overpy()

result = manager.get_query_result(
    manager.get_query(
        area="Волгоград",
        filters=[
            manager.node_by_name("Магнит", exact=False),
        ],
    ),
)

print(result)
