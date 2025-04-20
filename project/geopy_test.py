###  ==== Тестирование библиотеки geopy ==== ###
from geopy.geocoders import Nominatim


# Инициализация апи геолокатора.
geolocator = Nominatim(user_agent="test_agent")

# --- Тыкаемся в город
# Преобразуем адрес города в координаты
location = geolocator.geocode("Волгоград, Россия")
latitude, longitude = location.latitude, location.longitude
# 48.7081906, 44.5153353
print(f"Координаты: {latitude}, {longitude}")

# Обратное геокодирование
reverse_location = geolocator.reverse(f'{latitude}, {longitude}')
# Памятник защитникам Красного Царицына и Сталинграда,
# улица Аллея Героев, Центральный район, Волгоград,
# городской округ Волгоград, Волгоградская область, 400066, Россия
print(f"Адрес: {reverse_location.address}")

# --- Достопримечательность
# Мамаев курган, Волгоград, городской округ Волгоград,
# Волгоградская область, 400078, Россия
location = geolocator.geocode('Мамаев курган')
print(location.address)

# --- Мой дом!
# переулок Ногина, пос. Линейный, Тракторозаводский район, Волгоград,
# городской округ Волгоград, Волгоградская область, 400088, Россия
location = geolocator.geocode('Тракторозаводский район, переулок Ногина, д. 34')
print(location.address)

# --- Локализация
# Wolgograd, Stadtkreis Wolgograd, Oblast Wolgograd, Föderationskreis Südrussland, Russland
location = geolocator.geocode("Волгоград", language="de")
print(location.address)
# Volgograd, Volgograd Oblast, Southern Federal District, Russia
location = geolocator.geocode("Волгоград", language="en")
print(location.address)

# --- Находим все совпадения
# City of New York, New York, United States
# Нью-Йорк, Торецька міська громада, Бахмутський район, Донецька область, 85294-85297, Україна
# New York, United States
locations = geolocator.geocode("Нью Йорк", exactly_one=False)

for location in locations:
    print(location.address)


###  ==== Тестирование библиотеки osmnx ==== ###
