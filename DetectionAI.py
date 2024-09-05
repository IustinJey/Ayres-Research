from enum import Enum

class VehicleType(Enum):
    CAR = 'Car',
    TRUCK = 'Truck',
    EMERGENCY = 'Emergency'
    UNKNOWN = 'Unknown'


class Orientation(Enum):
    FRONT = 'Front',
    REAR = 'Rear'
    UNKNOWN = 'Unknown'


class DetectionAI:
    def __init__(self):
        self.__license_plate = 'Unknown'
        self.__vehicle_type = VehicleType.UNKNOWN
        self.__orientation = Orientation.UNKNOWN

    def get_license_plate(self) -> str:
        return self.__license_plate

    def set_license_plate(self, license_plate: str):
        self.__license_plate = license_plate

    def set_car(self):
        self.__vehicle_type = VehicleType.CAR

    def set_truck(self):
        self.__vehicle_type = VehicleType.TRUCK

    def set_emergency(self):
        self.__vehicle_type = VehicleType.EMERGENCY

    def set_rear(self):
        self.__orientation = Orientation.REAR

    def set_front(self):
        self.__orientation = Orientation.FRONT

    def is_car(self):
        return self.__vehicle_type == VehicleType.CAR

    def is_truck(self):
        return self.__vehicle_type == VehicleType.TRUCK

    def is_emergency(self):
        return self.__vehicle_type == VehicleType.EMERGENCY

    def is_front(self):
        return self.__orientation == Orientation.FRONT

    def is_rear(self):
        return self.__orientation == Orientation.REAR

    def is_vehicle_unknown(self):
        return self.__vehicle_type == VehicleType.UNKNOWN

    def is_orientation_unknown(self):
        return self.__orientation == Orientation.UNKNOWN

    def __str__(self):
        return f'Vehicle Type: {self.__vehicle_type.name}, Orientation: {self.__orientation.name}, License Plate: {self.__license_plate}'

