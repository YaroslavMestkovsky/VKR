from geopy.geocoders import Nominatim


class Geopy:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="test_agent")

    def _geocode(self, loc_name, exactly_one=True):
        return self.geolocator.geocode(loc_name, exactly_one=exactly_one)

    def address(self, loc_name, exactly_one=True):
        location = self._geocode(loc_name, exactly_one)

        return location.address if location else None

    def coordinates(self, loc_name, exactly_one=True):
        location = self._geocode(loc_name, exactly_one)

        return location.latitude, location.longitude if location else None

    def reverse(self, latitude, longitude):
        return self.geolocator.reverse((latitude, longitude))
