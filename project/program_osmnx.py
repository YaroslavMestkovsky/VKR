import osmnx as ox
import matplotlib.pyplot as plt

# 1. Загрузка данных
pharmacy_tag = {"amenity": "pharmacy"}
school_tag = {"amenity": "school"}
shop_tag = {"shop": ["supermarket", "convenience"]}
traffic_signals_tag = {"highway": "traffic_signals"}
house_tag = {"building": True} #, "building:levels": lambda x: int(x) >= 9} # дома от 9 этажей и выше

place_name = "Волгоград"

# Загрузка геометрии
pharmacy = ox.features_from_place(place_name, pharmacy_tag)
schools = ox.features_from_place(place_name, school_tag)
shops = ox.features_from_place(place_name, shop_tag)
traffic_signals = ox.features_from_place(place_name, traffic_signals_tag)
houses = ox.features_from_place(place_name, house_tag)

# 2. Визуализация
fig, ax = plt.subplots(figsize=(80, 80))
ax.set_xlim(44.4, 44.7)
ax.set_ylim(48.6, 48.8)

# Дорожная сеть для контекста
G = ox.graph_from_place(place_name, network_type="drive")
ox.plot_graph(G, ax=ax, node_size=0, edge_color="gray", show=False)

houses.plot(ax=ax, color="lightblue", markersize=0.5, label="Дома")
pharmacy.plot(ax=ax, color="red", markersize=0.5, label="Аптеки")
schools.plot(ax=ax, color="brown", markersize=0.5, label="Школы")
shops.plot(ax=ax, color="green", markersize=0.5, label="Магазины")
traffic_signals.plot(ax=ax, color="yellow", markersize=0.5, label="Светофоры")

plt.title("Волгоград")
plt.legend(fontsize=16)
plt.show()
