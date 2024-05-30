COLORS = {
    'none': (140, 140, 140),           # medium grey
    'default_bright': (255, 55, 139),  # bright pink
    'default_dark': (64, 17, 159),     # dark purple
    'true': (255, 80, 80),             # green
    'false': (125, 215, 82),           # red
    'accessible_true': (),             # TODO
    'accessible_false': (),            # TODO
    'black': (0, 0, 0),                # black
    'white': (255, 255, 255),          # white
}

DEFAULT_COLOR_MAPPING = {
    'default': ('default_bright', 'default_dark'),
    'bool': ('true', 'false')
}