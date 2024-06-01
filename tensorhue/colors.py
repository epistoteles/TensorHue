from rich.color_triplet import ColorTriplet

COLORS = {
    "none": ColorTriplet(140, 140, 140),  # medium grey
    "default_bright": ColorTriplet(255, 55, 139),  # bright pink
    "default_dark": ColorTriplet(64, 17, 159),  # dark purple
    "true": ColorTriplet(255, 80, 80),  # green
    "false": ColorTriplet(125, 215, 82),  # red
    "accessible_true": ColorTriplet(255, 80, 80),  # TODO
    "accessible_false": ColorTriplet(125, 215, 82),  # TODO
    "black": ColorTriplet(0, 0, 0),  # black
    "white": ColorTriplet(255, 255, 255),  # white
}

COLOR_MAPPING = {  # pylint: disable=consider-using-namedtuple-or-dataclass
    "default": ("default_dark", "default_bright"),
    "bool": ("false", "true"),
}
