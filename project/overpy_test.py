###  ==== Тестирование библиотеки overpy ==== ###
import overpy


# Инициализация апи overpy.
api = overpy.Overpass()

# --- Поиск точек интереса.
# Запрос: найти все Магниты в Волгограде
# query = """
#     [out:json];
#     area["name"="Волгоград"]->.searchArea;
#     (
#         node["name"~"Магнит", i](area.searchArea);
#     );
#     out body;
# """
#
# result = api.query(query)
#
# for node in result.nodes[:5]:
#     print(
#         f'''
#             brand: {node.tags.get("brand")}
#             brand:wikidata: {node.tags.get("brand:wikidata")}
#             brand:wikipedia: {node.tags.get("brand:wikipedia")}
#             contact:phone: {node.tags.get("contact:phone")}
#             contact:website: {node.tags.get("contact:website")}
#             int_name: {node.tags.get("int_name")}
#             name: {node.tags.get("name")}
#             name:en: {node.tags.get("name:en")}
#             name:ru: {node.tags.get("name:ru")}
#             opening_hours: {node.tags.get("opening_hours")}
#             operator: {node.tags.get("operator")}
#             shop: {node.tags.get("shop")}
#         '''
#     )

# Вывод:

# brand: Магнит
# brand:wikidata: Q940518
# brand:wikipedia: ru:Магнит (сеть магазинов)
# contact:phone: +7 800 2009002
# contact:website: http://magnit-info.ru
# int_name: Magnit
# name: Магнит
# name:en: Magnit
# name:ru: Магнит
# opening_hours: 08:30-22:00
# operator: АО Тандер
# shop: supermarket

# --- Поиск всех аптек! в Волгограде
# query = """
# [out:json];
# area["name"="Волгоград"]->.searchArea;
# (
#     node["amenity"="pharmacy"](area.searchArea);
# );
# out body;
# """
#
# result = api.query(query)
#
# print(f"Найдено аптек: {len(result.nodes)}")
#
# for node in result.nodes[:5]:
#     print(f"Аптека: {node.tags.get('name')} ({node.lat}, {node.lon})")


# --- Различные запросы.
# Найти все здания в заданной области:
# query = """
# [out:json];
# (
#     way["building"](55.70, 37.50, 55.80, 37.60);
# );
# out body;
# """

# Найти все главные дороги в городе:
query = """
[out:json];
area["name"="Волгоград"]->.searchArea;
(
    way["highway"="primary"](area.searchArea);
);
out body;
"""

# Найти все парки и стадионы:
query = """
[out:json];
area["name"="New York"]->.searchArea;
(
    node["leisure"="park"](area.searchArea);
    node["leisure"="stadium"](area.searchArea);
);
out body;
"""