"""
PySide6 3D Color Wheel — Studio Edition
Professional color workstation with input, gradients, image palettes, and more.

Requirements:
    pip install PySide6 numpy PyOpenGL Pillow

Run:
    python color_wheel_studio.py
"""

import sys
import math
import ctypes
import json
import os
from datetime import datetime
from collections import deque

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QTabWidget, QSlider,
    QSpinBox, QMenu, QMenuBar, QStatusBar, QFileDialog, QMessageBox,
    QScrollArea, QComboBox, QLineEdit, QCheckBox, QToolButton,
    QUndoStack, QUndoCommand, QDrag, QSizePolicy
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import (
    Qt, QTimer, QPointF, Signal, QSize, QSettings, QMimeData, QUrl
)
from PySide6.QtGui import (
    QSurfaceFormat, QColor, QPainter, QPen, QFont, QLinearGradient,
    QRadialGradient, QConicalGradient, QGuiApplication, QImage,
    QKeySequence, QAction, QPixmap, QDragEnterEvent, QDropEvent,
    QBrush, QPalette
)


# ============================================================================
# COLOR UTILITIES
# ============================================================================
class ColorUtils:
    @staticmethod
    def rgb_to_hsl(r, g, b):
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        mx, mn = max(r, g, b), min(r, g, b)
        l = (mx + mn) / 2.0
        if mx == mn:
            h = s = 0.0
        else:
            d = mx - mn
            s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
            if mx == r: h = (g - b) / d + (6.0 if g < b else 0.0)
            elif mx == g: h = (b - r) / d + 2.0
            else: h = (r - g) / d + 4.0
            h /= 6.0
        return h, s, l

    @staticmethod
    def hsl_to_rgb(h, s, l):
        if s == 0:
            v = int(l * 255)
            return v, v, v
        def hue2rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1/3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3)
        return int(r*255), int(g*255), int(b*255)

    @staticmethod
    def rgb_to_hsv(r, g, b):
        r, g, b = r/255.0, g/255.0, b/255.0
        mx, mn = max(r, g, b), min(r, g, b)
        d = mx - mn
        v = mx
        s = 0 if mx == 0 else d / mx
        if mx == mn: h = 0
        else:
            if mx == r: h = (g - b) / d + (6 if g < b else 0)
            elif mx == g: h = (b - r) / d + 2
            else: h = (r - g) / d + 4
            h /= 6
        return h, s, v

    @staticmethod
    def hsv_to_rgb(h, s, v):
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i %= 6
        if i == 0: r, g, b = v, t, p
        elif i == 1: r, g, b = q, v, p
        elif i == 2: r, g, b = p, v, t
        elif i == 3: r, g, b = p, q, v
        elif i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q
        return int(r*255), int(g*255), int(b*255)

    @staticmethod
    def rgb_to_hex(r, g, b):
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def hex_to_rgb(hex_str):
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) == 3:
            hex_str = ''.join(c*2 for c in hex_str)
        if len(hex_str) != 6:
            return None
        try:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return None

    @staticmethod
    def relative_luminance(rgb):
        def lin(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = [lin(x) for x in rgb]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @staticmethod
    def contrast_ratio(c1, c2):
        l1 = ColorUtils.relative_luminance(c1)
        l2 = ColorUtils.relative_luminance(c2)
        lighter, darker = max(l1, l2), min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    @staticmethod
    def apca_contrast(text, bg):
        """APCA (WCAG 3) contrast score. Returns Lc value."""
        def y_s(rgb):
            r, g, b = [x / 255.0 for x in rgb]
            r = r ** 2.2 if r > 0.03928 else r / 12.92
            g = g ** 2.2 if g > 0.03928 else g / 12.92
            b = b ** 2.2 if b > 0.03928 else b / 12.92
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        y_text = y_s(text)
        y_bg = y_s(bg)
        if y_bg > y_text:
            lc = (y_bg ** 0.56 - y_text ** 0.57) * 1.14
        else:
            lc = (y_bg ** 1.05 - y_text ** 0.80) * 1.14
        return abs(lc) * 100

    @staticmethod
    def apca_rating(lc, is_body_text=True, font_size=16, font_weight=400):
        """Return APCA rating based on Lc and text properties."""
        if is_body_text and font_weight < 700 and font_size < 24:
            threshold = 75
        elif font_weight >= 700 or font_size >= 24:
            threshold = 60
        else:
            threshold = 75
        if lc >= 90: return "Platinum"
        if lc >= 75: return "Silver"
        if lc >= threshold: return "Pass"
        if lc >= 45: return "Marginal"
        return "Fail"

    @staticmethod
    def wcag_rating(ratio):
        if ratio >= 7: return "AAA"
        if ratio >= 4.5: return "AA"
        if ratio >= 3: return "AA Large"
        return "Fail"

    @staticmethod
    def simulate_colorblindness(rgb, mode):
        matrices = {
            'protanopia':    [[0.567,0.433,0.000],[0.558,0.442,0.000],[0.000,0.242,0.758]],
            'deuteranopia':  [[0.625,0.375,0.000],[0.700,0.300,0.000],[0.000,0.300,0.700]],
            'tritanopia':    [[0.950,0.050,0.000],[0.000,0.433,0.567],[0.000,0.475,0.525]],
            'achromatopsia': [[0.299,0.587,0.114],[0.299,0.587,0.114],[0.299,0.587,0.114]],
        }
        m = matrices.get(mode, np.eye(3))
        r, g, b = rgb
        nr = max(0, min(255, int(m[0][0]*r + m[0][1]*g + m[0][2]*b)))
        ng = max(0, min(255, int(m[1][0]*r + m[1][1]*g + m[1][2]*b)))
        nb = max(0, min(255, int(m[2][0]*r + m[2][1]*g + m[2][2]*b)))
        return (nr, ng, nb)

    @staticmethod
    def complementary(rgb):
        h, s, l = ColorUtils.rgb_to_hsl(*rgb)
        return ColorUtils.hsl_to_rgb((h + 0.5) % 1.0, s, l)

    @staticmethod
    def analogous(rgb, angle=30):
        h, s, l = ColorUtils.rgb_to_hsl(*rgb)
        return [
            ColorUtils.hsl_to_rgb((h - angle/360) % 1.0, s, l),
            rgb,
            ColorUtils.hsl_to_rgb((h + angle/360) % 1.0, s, l),
        ]

    @staticmethod
    def triadic(rgb):
        h, s, l = ColorUtils.rgb_to_hsl(*rgb)
        return [rgb,
                ColorUtils.hsl_to_rgb((h + 1/3) % 1.0, s, l),
                ColorUtils.hsl_to_rgb((h + 2/3) % 1.0, s, l)]

    @staticmethod
    def tetradic(rgb):
        h, s, l = ColorUtils.rgb_to_hsl(*rgb)
        return [rgb,
                ColorUtils.hsl_to_rgb((h + 0.25) % 1.0, s, l),
                ColorUtils.hsl_to_rgb((h + 0.5) % 1.0, s, l),
                ColorUtils.hsl_to_rgb((h + 0.75) % 1.0, s, l)]

    @staticmethod
    def split_complementary(rgb):
        h, s, l = ColorUtils.rgb_to_hsl(*rgb)
        return [rgb,
                ColorUtils.hsl_to_rgb((h + 150/360) % 1.0, s, l),
                ColorUtils.hsl_to_rgb((h + 210/360) % 1.0, s, l)]

    @staticmethod
    def monochromatic(rgb, steps=5):
        h, s, l = ColorUtils.rgb_to_hsl(*rgb)
        return [ColorUtils.hsl_to_rgb(h, s, 0.15 + 0.7 * (i / (steps - 1)))
                for i in range(steps)]

    @staticmethod
    def shades(rgb, steps=7):
        return [(int(rgb[0]*(1-i/(steps-1))), int(rgb[1]*(1-i/(steps-1))),
                 int(rgb[2]*(1-i/(steps-1)))) for i in range(steps)]

    @staticmethod
    def tints(rgb, steps=7):
        return [(int(rgb[0]+(255-rgb[0])*i/(steps-1)),
                 int(rgb[1]+(255-rgb[1])*i/(steps-1)),
                 int(rgb[2]+(255-rgb[2])*i/(steps-1))) for i in range(steps)]

    @staticmethod
    def tones(rgb, steps=7):
        return [(int(rgb[0]+(128-rgb[0])*i/(steps-1)),
                 int(rgb[1]+(128-rgb[1])*i/(steps-1)),
                 int(rgb[2]+(128-rgb[2])*i/(steps-1))) for i in range(steps)]

    @staticmethod
    def mix_colors(c1, c2, ratio=0.5):
        return (int(c1[0]*(1-ratio)+c2[0]*ratio),
                int(c1[1]*(1-ratio)+c2[1]*ratio),
                int(c1[2]*(1-ratio)+c2[2]*ratio))

    @staticmethod
    def random_color():
        import random
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    @staticmethod
    def kelvin_to_rgb(kelvin):
        """Convert color temperature (K) to RGB. Based on Tanner Helland's algorithm."""
        temp = kelvin / 100.0
        if temp <= 66:
            r = 255
            g = max(0, min(255, int(99.4708 * math.log(temp) - 161.1196)))
            if temp <= 19: b = 0
            else: b = max(0, min(255, int(138.5177 * math.log(temp - 10) - 305.0448)))
        else:
            r = max(0, min(255, int(329.6987 * ((temp - 60) ** -0.13320))))
            g = max(0, min(255, int(288.1222 * ((temp - 60) ** -0.07551))))
            b = 255
        return (r, g, b)

    @staticmethod
    def median_cut(pixels, max_colors=8):
        """Extract dominant colors using median cut algorithm."""
        if not pixels:
            return []
        buckets = [list(pixels)]
        while len(buckets) < max_colors:
            # Find bucket with largest range
            best_idx = -1
            best_range = -1
            best_channel = 0
            for i, bucket in enumerate(buckets):
                if len(bucket) < 2:
                    continue
                for ch in range(3):
                    vals = [p[ch] for p in bucket]
                    rng = max(vals) - min(vals)
                    if rng > best_range:
                        best_range = rng
                        best_idx = i
                        best_channel = ch
            if best_idx < 0:
                break
            bucket = buckets[best_idx]
            bucket.sort(key=lambda p: p[best_channel])
            mid = len(bucket) // 2
            buckets[best_idx] = bucket[:mid]
            buckets.append(bucket[mid:])
        # Average each bucket
        result = []
        for bucket in buckets:
            if not bucket:
                continue
            avg = tuple(int(sum(p[ch] for p in bucket) / len(bucket)) for ch in range(3))
            result.append(avg)
        # Sort by luminance (brightest first)
        result.sort(key=lambda c: ColorUtils.relative_luminance(c), reverse=True)
        return result


# ============================================================================
# CSS NAMED COLORS (148 colors)
# ============================================================================
CSS_NAMED_COLORS = {
    "aliceblue": (240,248,255), "antiquewhite": (250,235,215), "aqua": (0,255,255),
    "aquamarine": (127,255,212), "azure": (240,255,255), "beige": (245,245,220),
    "bisque": (255,228,196), "black": (0,0,0), "blanchedalmond": (255,235,205),
    "blue": (0,0,255), "blueviolet": (138,43,226), "brown": (165,42,42),
    "burlywood": (222,184,135), "cadetblue": (95,158,160), "chartreuse": (127,255,0),
    "chocolate": (210,105,30), "coral": (255,127,80), "cornflowerblue": (100,149,237),
    "cornsilk": (255,248,220), "crimson": (220,20,60), "cyan": (0,255,255),
    "darkblue": (0,0,139), "darkcyan": (0,139,139), "darkgoldenrod": (184,134,11),
    "darkgray": (169,169,169), "darkgreen": (0,100,0), "darkkhaki": (189,183,107),
    "darkmagenta": (139,0,139), "darkolivegreen": (85,107,47), "darkorange": (255,140,0),
    "darkorchid": (153,50,204), "darkred": (139,0,0), "darksalmon": (233,150,122),
    "darkseagreen": (143,188,143), "darkslateblue": (72,61,139), "darkslategray": (47,79,79),
    "darkturquoise": (0,206,209), "darkviolet": (148,0,211), "deeppink": (255,20,147),
    "deepskyblue": (0,191,255), "dimgray": (105,105,105), "dodgerblue": (30,144,255),
    "firebrick": (178,34,34), "floralwhite": (255,250,240), "forestgreen": (34,139,34),
    "fuchsia": (255,0,255), "gainsboro": (220,220,220), "ghostwhite": (248,248,255),
    "gold": (255,215,0), "goldenrod": (218,165,32), "gray": (128,128,128),
    "green": (0,128,0), "greenyellow": (173,255,47), "honeydew": (240,255,240),
    "hotpink": (255,105,180), "indianred": (205,92,92), "indigo": (75,0,130),
    "ivory": (255,255,240), "khaki": (240,230,140), "lavender": (230,230,250),
    "lavenderblush": (255,240,245), "lawngreen": (124,252,0), "lemonchiffon": (255,250,205),
    "lightblue": (173,216,230), "lightcoral": (240,128,128), "lightcyan": (224,255,255),
    "lightgoldenrodyellow": (250,250,210), "lightgray": (211,211,211),
    "lightgreen": (144,238,144), "lightpink": (255,182,193), "lightsalmon": (255,160,122),
    "lightseagreen": (32,178,170), "lightskyblue": (135,206,250),
    "lightslategray": (119,136,153), "lightsteelblue": (176,196,222),
    "lightyellow": (255,255,224), "lime": (0,255,0), "limegreen": (50,205,50),
    "linen": (250,240,230), "magenta": (255,0,255), "maroon": (128,0,0),
    "mediumaquamarine": (102,205,170), "mediumblue": (0,0,205),
    "mediumorchid": (186,85,211), "mediumpurple": (147,112,219),
    "mediumseagreen": (60,179,113), "mediumslateblue": (123,104,238),
    "mediumspringgreen": (0,250,154), "mediumturquoise": (72,209,204),
    "mediumvioletred": (199,21,133), "midnightblue": (25,25,112),
    "mintcream": (245,255,250), "mistyrose": (255,228,225), "moccasin": (255,228,181),
    "navajowhite": (255,222,173), "navy": (0,0,128), "oldlace": (253,245,230),
    "olive": (128,128,0), "olivedrab": (107,142,35), "orange": (255,165,0),
    "orangered": (255,69,0), "orchid": (218,112,214), "palegoldenrod": (238,232,170),
    "palegreen": (152,251,152), "paleturquoise": (175,238,238),
    "palevioletred": (219,112,147), "papayawhip": (255,239,213), "peachpuff": (255,218,185),
    "peru": (205,133,63), "pink": (255,192,203), "plum": (221,160,221),
    "powderblue": (176,224,230), "purple": (128,0,128), "rebeccapurple": (102,51,153),
    "red": (255,0,0), "rosybrown": (188,143,143), "royalblue": (65,105,225),
    "saddlebrown": (139,69,19), "salmon": (250,128,114), "sandybrown": (244,164,96),
    "seagreen": (46,139,87), "seashell": (255,245,238), "sienna": (160,82,45),
    "silver": (192,192,192), "skyblue": (135,206,235), "slateblue": (106,90,205),
    "slategray": (112,128,144), "snow": (255,250,250), "springgreen": (0,255,127),
    "steelblue": (70,130,180), "tan": (210,180,140), "teal": (0,128,128),
    "thistle": (216,191,216), "tomato": (255,99,71), "turquoise": (64,224,208),
    "violet": (238,130,238), "wheat": (245,222,179), "white": (255,255,255),
    "whitesmoke": (245,245,245), "yellow": (255,255,0), "yellowgreen": (154,205,50),
}


# ============================================================================
# UNDO COMMANDS
# ============================================================================
class ColorChangeCommand(QUndoCommand):
    def __init__(self, window, old_color, new_color, description="Change color"):
        super().__init__(description)
        self.window = window
        self.old_color = old_color
        self.new_color = new_color

    def redo(self):
        self.window._set_color_internal(self.new_color, record=False)

    def undo(self):
        self.window._set_color_internal(self.old_color, record=False)


# ============================================================================
# 3D COLOR WHEEL WIDGET
# ============================================================================
class ColorWheel3D(QOpenGLWidget):
    color_picked = Signal(tuple)
    color_hover = Signal(tuple)

    SHAPE_DISK, SHAPE_TORUS, SHAPE_SPHERE, SHAPE_CYLINDER = 0, 1, 2, 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rot_x = -25.0
        self.rot_y = 0.0
        self.zoom = -3.2
        self.shape = self.SHAPE_DISK
        self.auto_rotate = True
        self.wireframe = False
        self.colorblind_mode = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.start(16)
        self.time = 0.0
        self.fps = 0
        self._frame_count = 0
        self._fps_timer = 0.0
        self.last_mouse_pos = None
        self.mouse_pressed = False
        self.program_id = None
        self.vao_id = self.vbo_id = self.ibo_id = None
        self.num_indices = 0

        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(fmt)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def _animate(self):
        self.time += 0.016
        self._fps_timer += 0.016
        self._frame_count += 1
        if self._fps_timer >= 1.0:
            self.fps = self._frame_count
            self._frame_count = 0
            self._fps_timer = 0.0
        if self.auto_rotate and not self.mouse_pressed:
            self.rot_y += 0.25
        self.update()

    def set_shape(self, shape_id):
        if shape_id != self.shape:
            self.shape = shape_id
            if self.program_id:
                self._build_geometry()

    def set_wireframe(self, enabled):
        self.wireframe = enabled
        self.update()

    def set_colorblind_mode(self, mode):
        self.colorblind_mode = mode
        self.update()

    def initializeGL(self):
        from OpenGL.GL import (
            glClearColor, glEnable, GL_DEPTH_TEST, GL_BLEND,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_MULTISAMPLE
        )
        glClearColor(0.05, 0.05, 0.08, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_MULTISAMPLE)
        self._build_geometry()
        self._compile_shaders()

    def _build_geometry(self):
        from OpenGL.GL import (
            glGenVertexArrays, glBindVertexArray, glGenBuffers,
            glBindBuffer, GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER,
            glBufferData, GL_STATIC_DRAW, glVertexAttribPointer,
            glEnableVertexAttribArray, GL_FLOAT
        )
        if self.vao_id is None:
            self.vao_id = glGenVertexArrays(1)
            self.vbo_id = glGenBuffers(1)
            self.ibo_id = glGenBuffers(1)

        if self.shape == self.SHAPE_DISK: v, i = self._gen_disk()
        elif self.shape == self.SHAPE_TORUS: v, i = self._gen_torus()
        elif self.shape == self.SHAPE_SPHERE: v, i = self._gen_sphere()
        else: v, i = self._gen_cylinder()

        glBindVertexArray(self.vao_id)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        vdata = (ctypes.c_float * len(v))(*v)
        glBufferData(GL_ARRAY_BUFFER, len(v) * 4, vdata, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo_id)
        idata = (ctypes.c_uint * len(i))(*i)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(i) * 4, idata, GL_STATIC_DRAW)
        self.num_indices = len(i)

        stride = 8 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(3*4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, False, stride, ctypes.c_void_p(6*4))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)

    def _gen_disk(self):
        v, i = [], []
        rings, segs = 60, 120
        for ri in range(rings + 1):
            r = ri / rings
            for si in range(segs + 1):
                t = 2*math.pi*si/segs
                v.extend([r*math.cos(t), r*math.sin(t), 0, 0, 0, 1, si/segs, r])
        for ri in range(rings):
            for si in range(segs):
                f = ri*(segs+1)+si; s = f+segs+1
                i.extend([f, s, f+1, s, s+1, f+1])
        return v, i

    def _gen_torus(self):
        v, i = [], []
        R, r = 0.7, 0.32
        rings, segs = 48, 96
        for ri in range(rings + 1):
            u = ri/rings*2*math.pi
            for si in range(segs + 1):
                vv = si/segs*2*math.pi
                x = (R + r*math.cos(vv))*math.cos(u)
                y = (R + r*math.cos(vv))*math.sin(u)
                z = r*math.sin(vv)
                nx = math.cos(vv)*math.cos(u)
                ny = math.cos(vv)*math.sin(u)
                nz = math.sin(vv)
                v.extend([x, y, z, nx, ny, nz, ri/rings, si/segs])
        for ri in range(rings):
            for si in range(segs):
                f = ri*(segs+1)+si; s = f+segs+1
                i.extend([f, s, f+1, s, s+1, f+1])
        return v, i

    def _gen_sphere(self):
        v, i = [], []
        stacks, slices = 48, 96
        for si in range(stacks + 1):
            phi = math.pi*si/stacks
            for sl in range(slices + 1):
                th = 2*math.pi*sl/slices
                x = math.sin(phi)*math.cos(th)
                y = math.sin(phi)*math.sin(th)
                z = math.cos(phi)
                v.extend([x, y, z, x, y, z, sl/slices, 1.0 - si/stacks])
        for si in range(stacks):
            for sl in range(slices):
                f = si*(slices+1)+sl; s = f+slices+1
                i.extend([f, s, f+1, s, s+1, f+1])
        return v, i

    def _gen_cylinder(self):
        v, i = [], []
        segs, height = 96, 1.4
        for ri in range(2):
            y = -height/2 + ri*height
            for si in range(segs + 1):
                t = 2*math.pi*si/segs
                x, z = math.cos(t), math.sin(t)
                v.extend([x, y, z, x, 0, z, si/segs, ri])
        for si in range(segs):
            f = si; s = f+segs+1
            i.extend([f, s, f+1, s, s+1, f+1])
        offset = len(v)//8
        cb = offset
        v.extend([0, -height/2, 0, 0, -1, 0, 0.5, 0])
        for si in range(segs + 1):
            t = 2*math.pi*si/segs
            x, z = math.cos(t), math.sin(t)
            v.extend([x, -height/2, z, 0, -1, 0, 0.5+0.5*x, 0.5+0.5*z])
        for si in range(segs):
            i.extend([cb, cb+1+si+1, cb+1+si])
        offset = len(v)//8
        ct = offset
        v.extend([0, height/2, 0, 0, 1, 0, 0.5, 0])
        for si in range(segs + 1):
            t = 2*math.pi*si/segs
            x, z = math.cos(t), math.sin(t)
            v.extend([x, height/2, z, 0, 1, 0, 0.5+0.5*x, 0.5+0.5*z])
        for si in range(segs):
            i.extend([ct, ct+1+si, ct+1+si+1])
        return v, i

    def _compile_shaders(self):
        from OpenGL.GL import glDeleteShader
        from OpenGL.GL.shaders import compileProgram, compileShader

        vs_src = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec2 aUV;
        uniform mat4 uModel, uView, uProj;
        uniform float uTime;
        out vec3 vFragPos, vNormal;
        out vec2 vUV;
        void main() {
            vFragPos = vec3(uModel * vec4(aPos, 1.0));
            vNormal = normalize(mat3(transpose(inverse(uModel))) * aNormal);
            vUV = aUV;
            gl_Position = uProj * uView * vec4(vFragPos, 1.0);
        }
        """
        fs_src = """
        #version 330 core
        in vec3 vFragPos, vNormal;
        in vec2 vUV;
        out vec4 FragColor;
        uniform float uTime;
        uniform int uShape;
        uniform int uColorblind;

        vec3 hsl2rgb(float h, float s, float l) {
            float c = (1.0 - abs(2.0*l - 1.0)) * s;
            float hp = h * 6.0;
            float x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
            float m = l - c/2.0;
            vec3 rgb;
            int i = int(floor(hp));
            if (i == 0) rgb = vec3(c,x,0);
            else if (i == 1) rgb = vec3(x,c,0);
            else if (i == 2) rgb = vec3(0,c,x);
            else if (i == 3) rgb = vec3(0,x,c);
            else if (i == 4) rgb = vec3(x,0,c);
            else rgb = vec3(c,0,x);
            return rgb + m;
        }

        vec3 simulateCB(vec3 c, int mode) {
            if (mode == 0) return c;
            mat3 prot = mat3(0.567,0.558,0.000, 0.433,0.442,0.242, 0.000,0.000,0.758);
            mat3 deut = mat3(0.625,0.700,0.000, 0.375,0.300,0.300, 0.000,0.000,0.700);
            mat3 trit = mat3(0.950,0.000,0.000, 0.050,0.433,0.475, 0.000,0.567,0.525);
            mat3 achro = mat3(0.299,0.299,0.299, 0.587,0.587,0.587, 0.114,0.114,0.114);
            if (mode == 1) return prot * c;
            if (mode == 2) return deut * c;
            if (mode == 3) return trit * c;
            if (mode == 4) return achro * c;
            return c;
        }

        void main() {
            float hue, sat, light;
            if (uShape == 0) {
                hue = vUV.x; sat = clamp(vUV.y*1.3, 0.0, 1.0);
                light = 0.48 + 0.12*cos(vUV.y*3.14159);
            } else if (uShape == 1) {
                hue = vUV.x; sat = 0.95;
                light = 0.45 + 0.15*sin(vUV.y*6.28318);
            } else if (uShape == 2) {
                hue = vUV.x; sat = sin(vUV.y*3.14159); light = 0.5;
            } else {
                hue = vUV.x; sat = 0.95;
                light = 0.35 + 0.3*vUV.y;
            }
            vec3 baseColor = hsl2rgb(hue, sat, light);
            baseColor = simulateCB(baseColor, uColorblind);

            vec3 light1 = normalize(vec3(0.4, 0.8, 0.6));
            vec3 light2 = normalize(vec3(-0.6, -0.3, 0.5));
            vec3 norm = normalize(vNormal);
            float d1 = max(dot(norm, light1), 0.0);
            float d2 = max(dot(norm, light2), 0.0) * 0.4;
            vec3 viewDir = normalize(-vFragPos);
            float rim = pow(1.0 - max(dot(viewDir, norm), 0.0), 3.0) * 0.6;
            vec3 reflectDir = reflect(-light1, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 48.0);

            vec3 result = 0.22*baseColor + (d1+d2)*baseColor + 0.5*spec*vec3(1.0) + rim*baseColor*0.6;
            result += baseColor * 0.04 * (0.5 + 0.5*sin(uTime*1.5));
            FragColor = vec4(result, 1.0);
        }
        """
        try:
            vs = compileShader(vs_src, 0x8B31)
            fs = compileShader(fs_src, 0x8B30)
            self.program_id = compileProgram(vs, fs)
            glDeleteShader(vs); glDeleteShader(fs)
        except Exception as e:
            print(f"Shader error: {e}")
            raise

    def paintGL(self):
        from OpenGL.GL import (
            glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
            glUseProgram, glBindVertexArray, glDrawElements,
            GL_TRIANGLES, GL_UNSIGNED_INT,
            glGetUniformLocation, glUniformMatrix4fv, glUniform1f,
            glUniform1i, glViewport, glPolygonMode, GL_FRONT_AND_BACK,
            glLineWidth
        )
        w = max(1, self.width()); h = max(1, self.height())
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self.program_id: return

        glUseProgram(self.program_id)
        model = self._rotation_matrix(self.rot_x, self.rot_y, 0.0)
        view = np.eye(4, dtype=np.float32); view[3][2] = self.zoom
        proj = self._perspective_matrix(50.0, w/h, 0.1, 100.0)

        cb_map = {None: 0, 'protanopia': 1, 'deuteranopia': 2,
                  'tritanopia': 3, 'achromatopsia': 4}

        for name, val in [
            ("uModel", ('mat4', model)),
            ("uView", ('mat4', view)),
            ("uProj", ('mat4', proj)),
            ("uTime", ('float', self.time)),
            ("uShape", ('int', self.shape)),
            ("uColorblind", ('int', cb_map.get(self.colorblind_mode, 0))),
        ]:
            loc = glGetUniformLocation(self.program_id, name)
            if val[0] == 'mat4':
                glUniformMatrix4fv(loc, 1, False, val[1].ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            elif val[0] == 'float':
                glUniform1f(loc, val[1])
            else:
                glUniform1i(loc, val[1])

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINES)
            glLineWidth(1.0)

        glBindVertexArray(self.vao_id)
        glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_TRIANGLES)

    def _rotation_matrix(self, rx, ry, rz):
        cx, sx = np.cos(np.radians(rx)), np.sin(np.radians(rx))
        cy, sy = np.cos(np.radians(ry)), np.sin(np.radians(ry))
        Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype=np.float32)
        Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float32)
        return Ry @ Rx

    def _perspective_matrix(self, fov_deg, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
        p = np.zeros((4,4), dtype=np.float32)
        p[0][0] = f/aspect; p[1][1] = f
        p[2][2] = (far+near)/(near-far); p[2][3] = (2*far*near)/(near-far)
        p[3][2] = -1.0
        return p

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_pos = event.pos()
            self._do_pick(event.pos())

    def mouseMoveEvent(self, event):
        if self.mouse_pressed and self.last_mouse_pos:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            if abs(dx) + abs(dy) > 2:
                self.rot_y += dx * 0.4
                self.rot_x = max(-89, min(89, self.rot_x + dy * 0.4))
                self.last_mouse_pos = event.pos()
                self.update()
        else:
            self._do_pick(event.pos(), hover=True)

    def mouseReleaseEvent(self, event):
        self.mouse_pressed = False
        self.last_mouse_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom = max(-8.0, min(-1.5, self.zoom + delta * 0.005))
        self.update()

    def _do_pick(self, pos, hover=False):
        try:
            from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
            x = pos.x(); y = self.height() - pos.y() - 1
            if not (0 <= x < self.width() and 0 <= y < self.height()):
                return
            pixel = glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
            if pixel is None or len(pixel) < 3:
                return
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            if r + g + b < 30:
                return
            color = (r, g, b)
            if hover:
                self.color_hover.emit(color)
            else:
                self.color_picked.emit(color)
        except Exception:
            pass


# ============================================================================
# REUSABLE UI COMPONENTS
# ============================================================================
class ColorSwatch(QFrame):
    clicked = Signal(object)
    double_clicked = Signal(object)

    def __init__(self, color=(255,0,0), size=40, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptDrops(True)
        self._update_style()

    def set_color(self, color):
        self.color = color
        self._update_style()

    def _update_style(self):
        r, g, b = self.color
        self.setStyleSheet(f"""
            ColorSwatch {{
                background-color: rgb({r},{g},{b});
                border: 2px solid rgba(255,255,255,0.2);
                border-radius: 6px;
            }}
            ColorSwatch:hover {{ border: 2px solid #e94560; }}
        """)
        self.setToolTip(ColorUtils.rgb_to_hex(*self.color))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self, '_drag_start') and event.buttons() & Qt.LeftButton:
            if (event.pos() - self._drag_start).manhattanLength() > 10:
                drag = QDrag(self)
                mime = QMimeData()
                mime.setText(ColorUtils.rgb_to_hex(*self.color))
                mime.setData("application/x-color",
                             f"{self.color[0]},{self.color[1]},{self.color[2]}".encode())
                drag.setMimeData(mime)
                # Create preview pixmap
                pix = QPixmap(self.size())
                pix.fill(QColor(*self.color))
                drag.setPixmap(pix)
                drag.exec_(Qt.CopyAction)
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.color)
        super().mouseDoubleClickEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-color"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        data = bytes(event.mimeData().data("application/x-color")).decode()
        try:
            r, g, b = map(int, data.split(','))
            self.set_color((r, g, b))
            self.clicked.emit((r, g, b))
            event.acceptProposedAction()
        except Exception:
            pass


class ColorInput(QFrame):
    """Manual color input with HEX/RGB/HSL/HSV fields."""
    color_changed = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # HEX
        hex_row = QHBoxLayout()
        hex_row.addWidget(QLabel("HEX"))
        self.hex_input = QLineEdit()
        self.hex_input.setPlaceholderText("#FF5577")
        self.hex_input.setMaxLength(7)
        self.hex_input.textEdited.connect(self._on_hex_edited)
        hex_row.addWidget(self.hex_input)
        layout.addLayout(hex_row)

        # RGB
        rgb_row = QHBoxLayout()
        rgb_row.addWidget(QLabel("RGB"))
        self.r_spin = self._make_spin(0, 255)
        self.g_spin = self._make_spin(0, 255)
        self.b_spin = self._make_spin(0, 255)
        rgb_row.addWidget(self.r_spin); rgb_row.addWidget(self.g_spin); rgb_row.addWidget(self.b_spin)
        self.r_spin.valueChanged.connect(self._on_rgb_changed)
        self.g_spin.valueChanged.connect(self._on_rgb_changed)
        self.b_spin.valueChanged.connect(self._on_rgb_changed)
        layout.addLayout(rgb_row)

        # HSL
        hsl_row = QHBoxLayout()
        hsl_row.addWidget(QLabel("HSL"))
        self.h_spin = self._make_spin(0, 360)
        self.s_spin = self._make_spin(0, 100)
        self.l_spin = self._make_spin(0, 100)
        hsl_row.addWidget(self.h_spin); hsl_row.addWidget(self.s_spin); hsl_row.addWidget(self.l_spin)
        self.h_spin.valueChanged.connect(self._on_hsl_changed)
        self.s_spin.valueChanged.connect(self._on_hsl_changed)
        self.l_spin.valueChanged.connect(self._on_hsl_changed)
        layout.addLayout(hsl_row)

        # HSV
        hsv_row = QHBoxLayout()
        hsv_row.addWidget(QLabel("HSV"))
        self.hv_spin = self._make_spin(0, 360)
        self.sv_spin = self._make_spin(0, 100)
        self.vv_spin = self._make_spin(0, 100)
        hsv_row.addWidget(self.hv_spin); hsv_row.addWidget(self.sv_spin); hsv_row.addWidget(self.vv_spin)
        self.hv_spin.valueChanged.connect(self._on_hsv_changed)
        self.sv_spin.valueChanged.connect(self._on_hsv_changed)
        self.vv_spin.valueChanged.connect(self._on_hsv_changed)
        layout.addLayout(hsv_row)

        # Color picker button
        picker_btn = QPushButton("🎨 System Color Picker")
        picker_btn.clicked.connect(self._open_picker)
        layout.addWidget(picker_btn)

    def _make_spin(self, mn, mx):
        s = QSpinBox()
        s.setRange(mn, mx)
        s.setFixedWidth(70)
        return s

    def _on_hex_edited(self, text):
        if self._updating: return
        rgb = ColorUtils.hex_to_rgb(text)
        if rgb:
            self._updating = True
            self._set_rgb(*rgb)
            self._updating = False
            self.color_changed.emit(rgb)

    def _on_rgb_changed(self):
        if self._updating: return
        rgb = (self.r_spin.value(), self.g_spin.value(), self.b_spin.value())
        self._updating = True
        self.hex_input.setText(ColorUtils.rgb_to_hex(*rgb))
        self._set_hsl_from_rgb(*rgb)
        self._set_hsv_from_rgb(*rgb)
        self._updating = False
        self.color_changed.emit(rgb)

    def _on_hsl_changed(self):
        if self._updating: return
        h = self.h_spin.value() / 360.0
        s = self.s_spin.value() / 100.0
        l = self.l_spin.value() / 100.0
        rgb = ColorUtils.hsl_to_rgb(h, s, l)
        self._updating = True
        self._set_rgb(*rgb)
        self._set_hsv_from_rgb(*rgb)
        self._updating = False
        self.color_changed.emit(rgb)

    def _on_hsv_changed(self):
        if self._updating: return
        h = self.hv_spin.value() / 360.0
        s = self.sv_spin.value() / 100.0
        v = self.vv_spin.value() / 100.0
        rgb = ColorUtils.hsv_to_rgb(h, s, v)
        self._updating = True
        self._set_rgb(*rgb)
        self._set_hsl_from_rgb(*rgb)
        self._updating = False
        self.color_changed.emit(rgb)

    def _set_rgb(self, r, g, b):
        self.r_spin.setValue(r)
        self.g_spin.setValue(g)
        self.b_spin.setValue(b)
        self.hex_input.setText(ColorUtils.rgb_to_hex(r, g, b))

    def _set_hsl_from_rgb(self, r, g, b):
        h, s, l = ColorUtils.rgb_to_hsl(r, g, b)
        self.h_spin.setValue(int(h * 360))
        self.s_spin.setValue(int(s * 100))
        self.l_spin.setValue(int(l * 100))

    def _set_hsv_from_rgb(self, r, g, b):
        h, s, v = ColorUtils.rgb_to_hsv(r, g, b)
        self.hv_spin.setValue(int(h * 360))
        self.sv_spin.setValue(int(s * 100))
        self.vv_spin.setValue(int(v * 100))

    def set_color(self, rgb):
        self._updating = True
        self._set_rgb(*rgb)
        self._set_hsl_from_rgb(*rgb)
        self._set_hsv_from_rgb(*rgb)
        self._updating = False

    def _open_picker(self):
        from PySide6.QtWidgets import QColorDialog
        current = QColor(*self.r_spin.value(), self.g_spin.value(), self.b_spin.value())
        color = QColorDialog.getColor(current, self, "Pick Color")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self.set_color(rgb)
            self.color_changed.emit(rgb)


# ============================================================================
# GRADIENT GENERATOR
# ============================================================================
class GradientGenerator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stops = [(0.0, (233, 69, 96)), (1.0, (255, 107, 107))]
        self.gradient_type = "linear"  # linear, radial, angular
        self.angle = 90  # for linear
        self._build_ui()
        self._update_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Type selector
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Linear", "Radial", "Angular"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_row.addWidget(self.type_combo)
        type_row.addWidget(QLabel("Angle:"))
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(0, 360)
        self.angle_spin.setValue(90)
        self.angle_spin.valueChanged.connect(self._on_angle_changed)
        type_row.addWidget(self.angle_spin)
        type_row.addStretch()
        layout.addLayout(type_row)

        # Preview
        self.preview = QLabel()
        self.preview.setFixedHeight(80)
        self.preview.setStyleSheet("border-radius: 6px; border: 1px solid #333;")
        layout.addWidget(self.preview)

        # Stops
        self.stops_widget = QWidget()
        self.stops_layout = QVBoxLayout(self.stops_widget)
        self.stops_layout.setContentsMargins(0, 0, 0, 0)
        self.stops_layout.setSpacing(4)
        layout.addWidget(self.stops_widget)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Stop")
        add_btn.clicked.connect(self._add_stop)
        btn_row.addWidget(add_btn)
        self.copy_btn = QPushButton("📋 Copy CSS")
        self.copy_btn.setObjectName("primary")
        self.copy_btn.clicked.connect(self._copy_css)
        btn_row.addWidget(self.copy_btn)
        layout.addLayout(btn_row)

        self._rebuild_stops_ui()

    def _on_type_changed(self, text):
        self.gradient_type = text.lower()
        self._update_preview()

    def _on_angle_changed(self, val):
        self.angle = val
        self._update_preview()

    def _add_stop(self):
        if len(self.stops) >= 8:
            return
        # Insert in middle
        self.stops.sort(key=lambda s: s[0])
        new_pos = 0.5
        # Find gap
        for i in range(len(self.stops) - 1):
            if self.stops[i][0] < 0.5 < self.stops[i+1][0]:
                new_pos = 0.5
                break
        new_color = ColorUtils.mix_colors(self.stops[-2][1], self.stops[-1][1], 0.5)
        self.stops.append((new_pos, new_color))
        self.stops.sort(key=lambda s: s[0])
        self._rebuild_stops_ui()
        self._update_preview()

    def _rebuild_stops_ui(self):
        # Clear
        while self.stops_layout.count():
            item = self.stops_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.stop_rows = []
        for i, (pos, color) in enumerate(self.stops):
            row = QHBoxLayout()
            pos_slider = QSlider(Qt.Horizontal)
            pos_slider.setRange(0, 100)
            pos_slider.setValue(int(pos * 100))
            pos_slider.valueChanged.connect(lambda v, idx=i: self._on_pos_changed(idx, v))
            swatch = ColorSwatch(color, size=28)
            swatch.clicked.connect(lambda c, idx=i: self._on_color_changed(idx, c))
            remove_btn = QPushButton("×")
            remove_btn.setFixedSize(24, 24)
            remove_btn.setEnabled(len(self.stops) > 2)
            remove_btn.clicked.connect(lambda _, idx=i: self._remove_stop(idx))
            pos_lbl = QLabel(f"{int(pos*100)}%")
            pos_lbl.setFixedWidth(40)
            pos_slider.valueChanged.connect(lambda v, lbl=pos_lbl: lbl.setText(f"{v}%"))

            row.addWidget(pos_slider)
            row.addWidget(pos_lbl)
            row.addWidget(swatch)
            row.addWidget(remove_btn)
            container = QWidget()
            container.setLayout(row)
            self.stops_layout.addWidget(container)
            self.stop_rows.append((pos_slider, swatch, pos_lbl))

    def _on_pos_changed(self, idx, val):
        self.stops[idx] = (val / 100.0, self.stops[idx][1])
        self.stops.sort(key=lambda s: s[0])
        self._update_preview()

    def _on_color_changed(self, idx, color):
        self.stops[idx] = (self.stops[idx][0], color)
        self._update_preview()

    def _remove_stop(self, idx):
        if len(self.stops) <= 2:
            return
        self.stops.pop(idx)
        self._rebuild_stops_ui()
        self._update_preview()

    def _update_preview(self):
        w, h = self.preview.width() or 400, self.preview.height() or 80
        img = QImage(w, h, QImage.Format_RGB32)
        painter = QPainter(img)

        if self.gradient_type == "linear":
            angle_rad = math.radians(self.angle - 90)
            cx, cy = w/2, h/2
            dx = math.cos(angle_rad) * w
            dy = math.sin(angle_rad) * h
            grad = QLinearGradient(cx - dx/2, cy - dy/2, cx + dx/2, cy + dy/2)
        elif self.gradient_type == "radial":
            grad = QRadialGradient(w/2, h/2, max(w, h)/2)
        else:  # angular
            grad = QConicalGradient(w/2, h/2, -self.angle + 90)

        for pos, color in self.stops:
            grad.setColorAt(pos, QColor(*color))

        painter.fillRect(0, 0, w, h, grad)
        painter.end()
        self.preview.setPixmap(QPixmap.fromImage(img))

    def _copy_css(self):
        stops_str = ", ".join(
            f"{ColorUtils.rgb_to_hex(*c)} {int(p*100)}%"
            for p, c in sorted(self.stops, key=lambda s: s[0])
        )
        if self.gradient_type == "linear":
            css = f"linear-gradient({self.angle}deg, {stops_str})"
        elif self.gradient_type == "radial":
            css = f"radial-gradient(circle, {stops_str})"
        else:
            css = f"conic-gradient(from {self.angle}deg, {stops_str})"
        QGuiApplication.clipboard().setText(f"background: {css};")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()


# ============================================================================
# IMAGE PALETTE EXTRACTOR
# ============================================================================
class ImagePaletteExtractor(QWidget):
    palette_extracted = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.preview = QLabel("Drop an image here or click Load")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(200)
        self.preview.setStyleSheet("""
            QLabel {
                background: #0a0a18;
                border: 2px dashed #333;
                border-radius: 8px;
                color: #666;
                padding: 20px;
            }
        """)
        layout.addWidget(self.preview)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("📁 Load Image")
        load_btn.clicked.connect(self._load_image)
        btn_row.addWidget(load_btn)

        btn_row.addWidget(QLabel("Colors:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(3, 16)
        self.count_spin.setValue(8)
        btn_row.addWidget(self.count_spin)

        extract_btn = QPushButton("Extract Palette")
        extract_btn.setObjectName("primary")
        extract_btn.clicked.connect(self._extract)
        btn_row.addWidget(extract_btn)
        layout.addLayout(btn_row)

        self.palette_row = QHBoxLayout()
        self.palette_row.setSpacing(4)
        layout.addLayout(self.palette_row)

        self.current_image = None

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
        if path:
            self._set_image(path)

    def _set_image(self, path):
        pix = QPixmap(path)
        if pix.isNull():
            return
        self.current_image = path
        scaled = pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def _extract(self):
        if not self.current_image:
            return
        try:
            from PIL import Image
            img = Image.open(self.current_image).convert('RGB')
            # Resize for speed
            img.thumbnail((200, 200))
            pixels = list(img.getdata())
            count = self.count_spin.value()
            palette = ColorUtils.median_cut(pixels, count)

            # Clear existing
            while self.palette_row.count():
                item = self.palette_row.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            for color in palette:
                sw = ColorSwatch(color, size=36)
                sw.clicked.connect(lambda c=color: self.palette_extracted.emit([c]))
                self.palette_row.addWidget(sw)

            self.palette_extracted.emit(palette)
        except ImportError:
            QMessageBox.warning(self, "Missing Pillow",
                                "Install Pillow: pip install Pillow")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to extract: {e}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                self._set_image(path)


# ============================================================================
# NAMED COLORS BROWSER
# ============================================================================
class NamedColorsBrowser(QWidget):
    color_selected = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search color name...")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setSpacing(3)
        scroll.setWidget(self.grid_widget)
        layout.addWidget(scroll)

        self.swatches = []
        self._populate()

    def _populate(self, filter_text=""):
        # Clear
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.swatches.clear()

        filtered = [(name, rgb) for name, rgb in CSS_NAMED_COLORS.items()
                    if filter_text.lower() in name.lower()]
        filtered.sort(key=lambda x: x[0])

        cols = 4
        for i, (name, rgb) in enumerate(filtered):
            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setContentsMargins(2, 2, 2, 2)
            cl.setSpacing(2)
            sw = ColorSwatch(rgb, size=40)
            sw.clicked.connect(lambda c=rgb: self.color_selected.emit(c))
            lbl = QLabel(name)
            lbl.setStyleSheet("font-size: 10px; color: #aaa;")
            lbl.setAlignment(Qt.AlignCenter)
            cl.addWidget(sw)
            cl.addWidget(lbl)
            self.grid.addWidget(container, i // cols, i % cols)
            self.swatches.append(container)

    def _filter(self, text):
        self._populate(text)


# ============================================================================
# MAIN WINDOW
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 3D Color Wheel — Studio Edition")
        self.resize(1400, 880)

        self.current_color = (200, 80, 120)
        self.saved_colors = []
        self.max_saved = 20
        self.history = []
        self.max_history = 50
        self.secondary_color = (80, 120, 200)
        self.theme = "dark"

        # Undo stack
        self.undo_stack = QUndoStack(self)

        # Settings
        self.settings = QSettings("ColorWheelStudio", "Studio")
        self._load_session()

        self._build_menu()
        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()
        self._apply_theme()

        self._update_all()

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------
    def _load_session(self):
        try:
            colors_json = self.settings.value("saved_colors", "[]")
            self.saved_colors = [tuple(c) for c in json.loads(colors_json)]
            current = self.settings.value("current_color", None)
            if current:
                self.current_color = tuple(json.loads(current))
            theme = self.settings.value("theme", "dark")
            self.theme = theme
        except Exception:
            pass

    def _save_session(self):
        try:
            self.settings.setValue("saved_colors", json.dumps(self.saved_colors))
            self.settings.setValue("current_color", json.dumps(list(self.current_color)))
            self.settings.setValue("theme", self.theme)
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_session()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def _apply_theme(self):
        if self.theme == "dark":
            self.setStyleSheet("""
                QMainWindow { background-color: #0f0f1a; }
                QWidget { color: #e0e0e0; font-family: 'Segoe UI', Arial; font-size: 13px; }
                QFrame#panel {
                    background-color: #16162a; border: 1px solid #252545; border-radius: 10px;
                }
                QTabWidget::pane { border: 1px solid #252545; border-radius: 8px; background: #16162a; }
                QTabBar::tab {
                    background: #1a1a30; padding: 8px 14px; border: 1px solid #252545;
                    border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px;
                }
                QTabBar::tab:selected { background: #e94560; color: white; }
                QTabBar::tab:hover { background: #2a2a50; }
                QPushButton {
                    background-color: #1f1f3a; border: 1px solid #2f2f55;
                    padding: 7px 14px; border-radius: 6px; color: #e0e0e0;
                }
                QPushButton:hover { background-color: #2a2a50; border-color: #e94560; }
                QPushButton:checked { background-color: #e94560; border-color: #ff6b6b; color: white; }
                QPushButton#primary {
                    background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 #e94560, stop:1 #ff6b6b);
                    border: none; color: white; font-weight: bold;
                }
                QPushButton#primary:hover { background-color: #ff6b6b; }
                QLabel#title { font-size: 20px; font-weight: bold; color: #e94560; }
                QLabel#subtitle { color: #888; font-size: 12px; }
                QLabel#value {
                    font-family: 'Consolas', monospace; font-size: 13px; color: #a0e0ff;
                    background: #0a0a18; padding: 4px 8px; border-radius: 4px;
                }
                QLineEdit, QSpinBox, QComboBox {
                    background: #1f1f3a; border: 1px solid #2f2f55;
                    padding: 4px 8px; border-radius: 4px; color: #e0e0e0;
                }
                QSlider::groove:horizontal { height: 5px; background: #1f1f3a; border-radius: 3px; }
                QSlider::handle:horizontal {
                    width: 16px; margin: -5px 0; border-radius: 8px; background: #e94560;
                }
                QMenuBar { background: #0a0a15; color: #e0e0e0; }
                QMenuBar::item:selected { background: #e94560; }
                QMenu { background: #16162a; color: #e0e0e0; border: 1px solid #252545; }
                QMenu::item:selected { background: #e94560; }
                QStatusBar { background: #0a0a15; color: #888; }
                QToolBar { background: #0a0a15; border: none; spacing: 4px; padding: 2px; }
                QScrollArea { border: none; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #f0f0f5; }
                QWidget { color: #222; font-family: 'Segoe UI', Arial; font-size: 13px; }
                QFrame#panel {
                    background-color: #ffffff; border: 1px solid #d0d0e0; border-radius: 10px;
                }
                QTabWidget::pane { border: 1px solid #d0d0e0; border-radius: 8px; background: white; }
                QTabBar::tab {
                    background: #e8e8f0; padding: 8px 14px; border: 1px solid #d0d0e0;
                    border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px;
                }
                QTabBar::tab:selected { background: #e94560; color: white; }
                QPushButton {
                    background-color: #ffffff; border: 1px solid #c0c0d0;
                    padding: 7px 14px; border-radius: 6px;
                }
                QPushButton:hover { background-color: #f0f0ff; border-color: #e94560; }
                QPushButton:checked { background-color: #e94560; color: white; }
                QPushButton#primary {
                    background-color: #e94560; border: none; color: white; font-weight: bold;
                }
                QLabel#title { font-size: 20px; font-weight: bold; color: #e94560; }
                QLabel#subtitle { color: #666; font-size: 12px; }
                QLabel#value {
                    font-family: 'Consolas', monospace; font-size: 13px; color: #0066aa;
                    background: #f5f5ff; padding: 4px 8px; border-radius: 4px;
                }
                QLineEdit, QSpinBox, QComboBox {
                    background: white; border: 1px solid #c0c0d0; padding: 4px 8px; border-radius: 4px;
                }
                QSlider::groove:horizontal { height: 5px; background: #d0d0e0; border-radius: 3px; }
                QSlider::handle:horizontal { width: 16px; margin: -5px 0; border-radius: 8px; background: #e94560; }
                QMenuBar { background: #ffffff; color: #222; border-bottom: 1px solid #d0d0e0; }
                QMenuBar::item:selected { background: #e94560; color: white; }
                QMenu { background: white; color: #222; border: 1px solid #d0d0e0; }
                QMenu::item:selected { background: #e94560; color: white; }
                QStatusBar { background: #ffffff; color: #666; border-top: 1px solid #d0d0e0; }
                QToolBar { background: #ffffff; border: none; border-bottom: 1px solid #d0d0e0; spacing: 4px; }
                QScrollArea { border: none; }
            """)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.addAction("Export as &PNG...", self._export_png, QKeySequence("Ctrl+P"))
        file_menu.addAction("Export as &JSON...", self._export_json)
        file_menu.addAction("Export as &CSS...", self._export_css)
        file_menu.addAction("Export as &SCSS...", self._export_scss)
        file_menu.addAction("Export as &Tailwind config...", self._export_tailwind)
        file_menu.addAction("Export as &Swift...", self._export_swift)
        file_menu.addAction("Export as &Kotlin...", self._export_kotlin)
        file_menu.addAction("Export as &Android XML...", self._export_android)
        file_menu.addAction("Export as &SVG palette...", self._export_svg)
        file_menu.addSeparator()
        file_menu.addAction("&Import Palette (JSON)...", self._import_json)
        file_menu.addSeparator()
        file_menu.addAction("&Quit", self.close, QKeySequence("Ctrl+Q"))

        edit_menu = mb.addMenu("&Edit")
        undo_act = self.undo_stack.createUndoAction(self, "&Undo")
        undo_act.setShortcut(QKeySequence("Ctrl+Z"))
        edit_menu.addAction(undo_act)
        redo_act = self.undo_stack.createRedoAction(self, "&Redo")
        redo_act.setShortcut(QKeySequence("Ctrl+Y"))
        edit_menu.addAction(redo_act)
        edit_menu.addSeparator()
        edit_menu.addAction("&Copy HEX", lambda: self._copy(ColorUtils.rgb_to_hex(*self.current_color)),
                            QKeySequence("Ctrl+C"))

        view_menu = mb.addMenu("&View")
        self.act_fullscreen = view_menu.addAction("&Fullscreen", self._toggle_fullscreen, QKeySequence("F11"))
        self.act_fullscreen.setCheckable(True)
        view_menu.addSeparator()
        view_menu.addAction("&Reset View", self._reset_view, QKeySequence("Ctrl+R"))
        view_menu.addAction("Toggle &Wireframe", self._toggle_wireframe, QKeySequence("W"))
        view_menu.addSeparator()
        view_menu.addAction("Toggle &Theme", self._toggle_theme, QKeySequence("Ctrl+T"))

        tools_menu = mb.addMenu("&Tools")
        tools_menu.addAction("&Random Color", self._random_color, QKeySequence("Space"))
        tools_menu.addAction("Random &Harmonious", self._random_harmonious)
        tools_menu.addSeparator()
        cb_menu = tools_menu.addMenu("Color &Blindness")
        for name, mode in [("None", None), ("Protanopia", "protanopia"),
                            ("Deuteranopia", "deuteranopia"),
                            ("Tritanopia", "tritanopia"),
                            ("Achromatopsia", "achromatopsia")]:
            cb_menu.addAction(name, lambda m=mode: self._set_cb(m))

        help_menu = mb.addMenu("&Help")
        help_menu.addAction("&Shortcuts", self._show_shortcuts)
        help_menu.addAction("&About", self._show_about)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------
    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        for name, sid in [("Disk", 0), ("Torus", 1), ("Sphere", 2), ("Cylinder", 3)]:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, s=sid: self._set_shape(s))
            tb.addWidget(btn)
            if sid == 0:
                btn.setChecked(True)
                self._shape_btns = [btn]
            else:
                self._shape_btns.append(btn)

        tb.addSeparator()
        self.btn_auto = QPushButton("⏸ Auto")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(True)
        self.btn_auto.clicked.connect(self._toggle_auto)
        tb.addWidget(self.btn_auto)

        self.btn_wire = QPushButton("◻ Wire")
        self.btn_wire.setCheckable(True)
        self.btn_wire.clicked.connect(self._toggle_wireframe)
        tb.addWidget(self.btn_wire)

        tb.addSeparator()
        tb.addWidget(QLabel(" CB: "))
        self.cb_combo = QComboBox()
        self.cb_combo.addItems(["Normal", "Protanopia", "Deuteranopia", "Tritanopia", "Achromatopsia"])
        self.cb_combo.currentIndexChanged.connect(self._on_cb_combo)
        tb.addWidget(self.cb_combo)

        tb.addSeparator()
        btn_rand = QPushButton("🎲 Random")
        btn_rand.setObjectName("primary")
        btn_rand.clicked.connect(self._random_color)
        tb.addWidget(btn_rand)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def _build_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_lbl_shape = QLabel("Shape: Disk")
        self.status_lbl_fps = QLabel("FPS: 0")
        self.status_lbl_color = QLabel("Current: #C85078")
        self.status.addPermanentWidget(self.status_lbl_shape)
        self.status.addPermanentWidget(self.status_lbl_fps)
        self.status.addPermanentWidget(self.status_lbl_color)

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(250)

    def _update_status(self):
        shapes = ["Disk", "Torus", "Sphere", "Cylinder"]
        self.status_lbl_shape.setText(f"Shape: {shapes[self.wheel.shape]}")
        self.status_lbl_fps.setText(f"FPS: {self.wheel.fps}")
        self.status_lbl_color.setText(f"Current: {ColorUtils.rgb_to_hex(*self.current_color)}")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Left: 3D widget
        left = QVBoxLayout()
        left.setSpacing(6)

        hdr = QWidget()
        hl = QVBoxLayout(hdr); hl.setContentsMargins(0,0,0,0)
        title = QLabel("3D Color Wheel")
        title.setObjectName("title")
        sub = QLabel("Drag to rotate • Scroll to zoom • Click to pick • Drop images to extract palettes")
        sub.setObjectName("subtitle")
        hl.addWidget(title); hl.addWidget(sub)
        left.addWidget(hdr)

        self.wheel = ColorWheel3D()
        self.wheel.color_picked.connect(self._on_color_picked)
        self.wheel.color_hover.connect(self._on_color_hover)
        left.addWidget(self.wheel, stretch=1)
        root.addLayout(left, stretch=3)

        # Right: tabs
        right_tabs = QTabWidget()
        right_tabs.setFixedWidth(400)
        right_tabs.addTab(self._build_color_tab(), "🎨 Color")
        right_tabs.addTab(self._build_harmony_tab(), "🌈 Harmony")
        right_tabs.addTab(self._build_variations_tab(), "◐ Variations")
        right_tabs.addTab(self._build_gradient_tab(), "🌊 Gradient")
        right_tabs.addTab(self._build_explore_tab(), "🔍 Explore")
        right_tabs.addTab(self._build_palette_tab(), "💾 Palette")
        right_tabs.addTab(self._build_a11y_tab(), "♿ A11y")
        right_tabs.addTab(self._build_history_tab(), "📜 History")
        root.addWidget(right_tabs)

    # ------------------------------------------------------------------
    # Tab: Color (info + manual input + mixer)
    # ------------------------------------------------------------------
    def _build_color_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Big swatch
        self.big_swatch = ColorSwatch(self.current_color, size=160)
        self.big_swatch.setFixedSize(160, 160)
        sw_wrap = QWidget()
        swl = QHBoxLayout(sw_wrap); swl.setContentsMargins(0,0,0,0)
        swl.addWidget(self.big_swatch); swl.addStretch()
        layout.addWidget(sw_wrap)

        # Manual input
        input_frame = QFrame()
        input_frame.setObjectName("panel")
        ifl = QVBoxLayout(input_frame)
        ifl.setContentsMargins(10, 10, 10, 10)
        ifl.addWidget(QLabel("<b>Manual Input</b>"))
        self.color_input = ColorInput()
        self.color_input.color_changed.connect(self._on_manual_color_changed)
        ifl.addWidget(self.color_input)
        layout.addWidget(input_frame)

        # Values display
        for label_text, attr in [("HEX", "hex"), ("RGB", "rgb"), ("HSL", "hsl"), ("HSV", "hsv")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"<b>{label_text}</b>"))
            lbl = QLabel()
            lbl.setObjectName("value")
            row.addWidget(lbl, stretch=1)
            layout.addLayout(row)
            setattr(self, f"lbl_{attr}", lbl)

        # Color temperature
        temp_frame = QFrame()
        temp_frame.setObjectName("panel")
        tfl = QVBoxLayout(temp_frame)
        tfl.setContentsMargins(10, 10, 10, 10)
        tfl.addWidget(QLabel("<b>🌡️ Color Temperature</b>"))
        temp_row = QHBoxLayout()
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(1000, 15000)
        self.temp_slider.setValue(6500)
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        temp_row.addWidget(self.temp_slider)
        self.temp_lbl = QLabel("6500K")
        self.temp_lbl.setFixedWidth(60)
        temp_row.addWidget(self.temp_lbl)
        tfl.addLayout(temp_row)
        self.temp_swatch = ColorSwatch(ColorUtils.kelvin_to_rgb(6500), size=40)
        self.temp_swatch.clicked.connect(self._on_color_picked)
        tfl.addWidget(self.temp_swatch)
        layout.addWidget(temp_frame)

        # Mixer
        mix_frame = QFrame()
        mix_frame.setObjectName("panel")
        ml = QVBoxLayout(mix_frame)
        ml.setContentsMargins(10, 10, 10, 10)
        ml.addWidget(QLabel("<b>🎨 Color Mixer</b>"))
        mix_row = QHBoxLayout()
        self.mix_swatch1 = ColorSwatch(self.current_color, size=40)
        self.mix_swatch2 = ColorSwatch(self.secondary_color, size=40)
        self.mix_result = ColorSwatch(ColorUtils.mix_colors(self.current_color, self.secondary_color), size=40)
        mix_row.addWidget(self.mix_swatch1); mix_row.addStretch()
        mix_row.addWidget(self.mix_result); mix_row.addStretch()
        mix_row.addWidget(self.mix_swatch2)
        ml.addLayout(mix_row)
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("Ratio:"))
        self.mix_slider = QSlider(Qt.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(50)
        self.mix_slider.valueChanged.connect(self._update_mixer)
        ratio_row.addWidget(self.mix_slider)
        self.mix_ratio_lbl = QLabel("50%")
        self.mix_ratio_lbl.setFixedWidth(40)
        ratio_row.addWidget(self.mix_ratio_lbl)
        ml.addLayout(ratio_row)
        btn_row = QHBoxLayout()
        b1 = QPushButton("Use Current as A"); b1.clicked.connect(lambda: None)
        b2 = QPushButton("Use Current as B"); b2.clicked.connect(lambda: setattr(self, 'secondary_color', self.current_color) or self._update_mixer())
        btn_row.addWidget(b1); btn_row.addWidget(b2)
        ml.addLayout(btn_row)
        self.mix_result.clicked.connect(self._on_color_picked)
        layout.addWidget(mix_frame)

        # WCAG quick
        contrast_frame = QFrame()
        contrast_frame.setObjectName("panel")
        cfl = QVBoxLayout(contrast_frame)
        cfl.setContentsMargins(10, 10, 10, 10)
        cfl.addWidget(QLabel("<b>WCAG Contrast</b>"))
        self.lbl_contrast_white = QLabel()
        self.lbl_contrast_black = QLabel()
        cfl.addWidget(self.lbl_contrast_white)
        cfl.addWidget(self.lbl_contrast_black)
        layout.addWidget(contrast_frame)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_copy = QPushButton("📋 Copy HEX")
        btn_copy.setObjectName("primary")
        btn_copy.clicked.connect(lambda: self._copy(ColorUtils.rgb_to_hex(*self.current_color)))
        btn_row.addWidget(btn_copy)
        btn_save = QPushButton("💾 Save")
        btn_save.clicked.connect(self._save_current)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    # ------------------------------------------------------------------
    # Tab: Harmony
    # ------------------------------------------------------------------
    def _build_harmony_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.harmony_rows = {}
        self.harmony_sliders = {}

        # Analogous with angle control
        analogous_frame = self._make_harmony_frame("Analogous", "analogous")
        layout.addWidget(analogous_frame)

        for name, key in [
            ("Complementary", "complementary"),
            ("Triadic", "triadic"),
            ("Tetradic", "tetradic"),
            ("Split-Complementary", "split_complementary"),
            ("Monochromatic", "monochromatic"),
        ]:
            layout.addWidget(self._make_harmony_frame(name, key))

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _make_harmony_frame(self, name, key):
        frame = QFrame()
        frame.setObjectName("panel")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(10, 8, 10, 8)
        fl.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"<b>{name}</b>"))
        hdr.addStretch()
        fl.addLayout(hdr)

        # Angle slider for analogous
        if key == "analogous":
            angle_row = QHBoxLayout()
            angle_row.addWidget(QLabel("Angle:"))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(5, 90)
            slider.setValue(30)
            slider.valueChanged.connect(lambda v: self._update_all())
            angle_row.addWidget(slider)
            angle_lbl = QLabel("30°")
            angle_lbl.setFixedWidth(35)
            slider.valueChanged.connect(lambda v, lbl=angle_lbl: lbl.setText(f"{v}°"))
            angle_row.addWidget(angle_lbl)
            fl.addLayout(angle_row)
            self.harmony_sliders[key] = slider

        row = QHBoxLayout()
        row.setSpacing(4)
        swatches = []
        max_count = 5
        for i in range(max_count):
            sw = ColorSwatch((40, 40, 60), size=32)
            sw.clicked.connect(self._on_color_picked)
            row.addWidget(sw)
            swatches.append(sw)
        self.harmony_rows[key] = swatches
        fl.addLayout(row)

        return frame

    # ------------------------------------------------------------------
    # Tab: Variations (shades/tints/tones)
    # ------------------------------------------------------------------
    def _build_variations_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps:"))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(3, 15)
        self.spin_steps.setValue(7)
        self.spin_steps.valueChanged.connect(self._update_shades)
        steps_row.addWidget(self.spin_steps)
        steps_row.addStretch()
        layout.addLayout(steps_row)

        for name, attr in [("Shades (darker)", "shades"),
                            ("Tints (lighter)", "tints"),
                            ("Tones (muted)", "tones")]:
            frame = QFrame()
            frame.setObjectName("panel")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(10, 8, 10, 8)
            fl.setSpacing(4)
            fl.addWidget(QLabel(f"<b>{name}</b>"))
            row = QHBoxLayout()
            row.setSpacing(3)
            swatches = []
            for i in range(15):
                sw = ColorSwatch((40, 40, 60), size=22)
                sw.clicked.connect(self._on_color_picked)
                row.addWidget(sw)
                swatches.append(sw)
            setattr(self, f"shades_{attr}", swatches)
            fl.addLayout(row)
            layout.addWidget(frame)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    # ------------------------------------------------------------------
    # Tab: Gradient
    # ------------------------------------------------------------------
    def _build_gradient_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        self.gradient_gen = GradientGenerator()
        layout.addWidget(self.gradient_gen)
        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    # ------------------------------------------------------------------
    # Tab: Explore (image palette + named colors)
    # ------------------------------------------------------------------
    def _build_explore_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Image palette
        img_frame = QFrame()
        img_frame.setObjectName("panel")
        ifl = QVBoxLayout(img_frame)
        ifl.setContentsMargins(10, 10, 10, 10)
        ifl.addWidget(QLabel("<b>🖼️ Palette from Image</b>"))
        self.image_extractor = ImagePaletteExtractor()
        self.image_extractor.palette_extracted.connect(self._on_image_palette)
        ifl.addWidget(self.image_extractor)
        layout.addWidget(img_frame)

        # Named colors
        named_frame = QFrame()
        named_frame.setObjectName("panel")
        nfl = QVBoxLayout(named_frame)
        nfl.setContentsMargins(10, 10, 10, 10)
        nfl.addWidget(QLabel("<b>📚 CSS Named Colors</b>"))
        self.named_browser = NamedColorsBrowser()
        self.named_browser.color_selected.connect(self._on_color_picked)
        nfl.addWidget(self.named_browser)
        layout.addWidget(named_frame)

        scroll.setWidget(w)
        return scroll

    # ------------------------------------------------------------------
    # Tab: Palette
    # ------------------------------------------------------------------
    def _build_palette_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        pal_frame = QFrame()
        pal_frame.setObjectName("panel")
        pl = QVBoxLayout(pal_frame)
        pl.setContentsMargins(10, 10, 10, 10)
        plh = QHBoxLayout()
        plh.addWidget(QLabel("<b>Saved Palette</b>"))
        plh.addStretch()
        btn_clear = QPushButton("🗑 Clear")
        btn_clear.clicked.connect(self._clear_palette)
        plh.addWidget(btn_clear)
        pl.addLayout(plh)

        grid = QGridLayout()
        grid.setSpacing(4)
        self.palette_swatches = []
        for i in range(self.max_saved):
            sw = ColorSwatch((40, 40, 60), size=44)
            sw.clicked.connect(self._on_color_picked)
            sw.double_clicked.connect(lambda c, idx=i: self._remove_from_palette(idx))
            self.palette_swatches.append(sw)
            grid.addWidget(sw, i // 4, i % 4)
        pl.addLayout(grid)

        hint = QLabel("Click to load • Double-click to remove • Drag to reorder")
        hint.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        pl.addWidget(hint)
        layout.addWidget(pal_frame)

        # Export
        exp_frame = QFrame()
        exp_frame.setObjectName("panel")
        el = QVBoxLayout(exp_frame)
        el.setContentsMargins(10, 10, 10, 10)
        el.addWidget(QLabel("<b>📤 Export Palette</b>"))
        exp_grid = QGridLayout()
        for i, (label, func) in enumerate([
            ("PNG Image", self._export_png),
            ("JSON", self._export_json),
            ("CSS", self._export_css),
            ("SCSS", self._export_scss),
            ("Tailwind", self._export_tailwind),
            ("Swift", self._export_swift),
            ("Kotlin", self._export_kotlin),
            ("Android XML", self._export_android),
            ("SVG", self._export_svg),
        ]):
            btn = QPushButton(label)
            btn.clicked.connect(func)
            exp_grid.addWidget(btn, i // 3, i % 3)
        el.addLayout(exp_grid)
        layout.addWidget(exp_frame)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    # ------------------------------------------------------------------
    # Tab: Accessibility
    # ------------------------------------------------------------------
    def _build_a11y_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # CB preview
        cb_frame = QFrame()
        cb_frame.setObjectName("panel")
        cbl = QVBoxLayout(cb_frame)
        cbl.setContentsMargins(10, 10, 10, 10)
        cbl.addWidget(QLabel("<b>Color Blindness Simulation</b>"))
        self.cb_preview_rows = {}
        for name, mode in [("Protanopia", "protanopia"),
                            ("Deuteranopia", "deuteranopia"),
                            ("Tritanopia", "tritanopia"),
                            ("Achromatopsia", "achromatopsia")]:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(130)
            sw = ColorSwatch((40, 40, 60), size=32)
            hex_lbl = QLabel()
            hex_lbl.setStyleSheet("font-family: monospace; color: #888;")
            row.addWidget(lbl); row.addWidget(sw); row.addWidget(hex_lbl); row.addStretch()
            cbl.addLayout(row)
            self.cb_preview_rows[mode] = (sw, hex_lbl)
        layout.addWidget(cb_frame)

        # WCAG 2 contrast
        wcag_frame = QFrame()
        wcag_frame.setObjectName("panel")
        wcl = QVBoxLayout(wcag_frame)
        wcl.setContentsMargins(10, 10, 10, 10)
        wcl.addWidget(QLabel("<b>WCAG 2 Contrast</b>"))
        self.contrast_previews = []
        for bg_name, bg_color in [("White", (255, 255, 255)),
                                    ("Black", (0, 0, 0)),
                                    ("Dark", (30, 30, 50)),
                                    ("Light", (240, 240, 245))]:
            row = QHBoxLayout()
            preview = QLabel("Sample Aa 123")
            preview.setFixedHeight(36)
            preview.setAlignment(Qt.AlignCenter)
            preview.setStyleSheet("font-size: 15px; font-weight: bold; border-radius: 4px;")
            info = QLabel()
            info.setStyleSheet("font-family: monospace;")
            row.addWidget(preview, stretch=1); row.addWidget(info)
            wcl.addLayout(row)
            self.contrast_previews.append((preview, info, bg_color))
        layout.addWidget(wcag_frame)

        # APCA (WCAG 3)
        apca_frame = QFrame()
        apca_frame.setObjectName("panel")
        al = QVBoxLayout(apca_frame)
        al.setContentsMargins(10, 10, 10, 10)
        al.addWidget(QLabel("<b>APCA (WCAG 3) Contrast</b>"))
        al.addWidget(QLabel("New perceptual contrast model:",
                             "color: #888; font-size: 11px;"))
        self.apca_previews = []
        for bg_name, bg_color in [("White BG", (255, 255, 255)),
                                    ("Black BG", (0, 0, 0)),
                                    ("Dark BG", (30, 30, 50))]:
            row = QHBoxLayout()
            preview = QLabel("Sample Text Aa 123")
            preview.setFixedHeight(36)
            preview.setAlignment(Qt.AlignCenter)
            preview.setStyleSheet("font-size: 15px; font-weight: bold; border-radius: 4px;")
            info = QLabel()
            info.setStyleSheet("font-family: monospace;")
            row.addWidget(preview, stretch=1); row.addWidget(info)
            al.addLayout(row)
            self.apca_previews.append((preview, info, bg_color))
        layout.addWidget(apca_frame)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    # ------------------------------------------------------------------
    # Tab: History
    # ------------------------------------------------------------------
    def _build_history_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("<b>Recently Picked</b>"))
        hdr.addStretch()
        btn_clear = QPushButton("🗑 Clear")
        btn_clear.clicked.connect(self._clear_history)
        hdr.addWidget(btn_clear)
        layout.addLayout(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        hist_w = QWidget()
        self.history_layout = QVBoxLayout(hist_w)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(2)
        self.history_layout.addStretch()
        scroll.setWidget(hist_w)
        layout.addWidget(scroll)

        return w

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _set_shape(self, sid):
        self.wheel.set_shape(sid)
        for i, btn in enumerate(self._shape_btns):
            btn.setChecked(i == sid)

    def _toggle_auto(self):
        self.wheel.auto_rotate = self.btn_auto.isChecked()
        self.btn_auto.setText("⏸ Auto" if self.wheel.auto_rotate else "▶ Auto")

    def _toggle_wireframe(self):
        state = not self.wheel.wireframe
        self.wheel.set_wireframe(state)
        self.btn_wire.setChecked(state)

    def _reset_view(self):
        self.wheel.rot_x = -25.0
        self.wheel.rot_y = 0.0
        self.wheel.zoom = -3.2
        self.wheel.update()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.act_fullscreen.setChecked(False)
        else:
            self.showFullScreen()
            self.act_fullscreen.setChecked(True)

    def _toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self._apply_theme()

    def _set_cb(self, mode):
        self.wheel.set_colorblind_mode(mode)
        mode_to_idx = {None: 0, 'protanopia': 1, 'deuteranopia': 2,
                       'tritanopia': 3, 'achromatopsia': 4}
        self.cb_combo.blockSignals(True)
        self.cb_combo.setCurrentIndex(mode_to_idx.get(mode, 0))
        self.cb_combo.blockSignals(False)

    def _on_cb_combo(self, idx):
        modes = [None, 'protanopia', 'deuteranopia', 'tritanopia', 'achromatopsia']
        self._set_cb(modes[idx])

    def _random_color(self):
        self._on_color_picked(ColorUtils.random_color())

    def _random_harmonious(self):
        import random
        h = random.random()
        s = 0.6 + random.random() * 0.4
        l = 0.4 + random.random() * 0.3
        self._on_color_picked(ColorUtils.hsl_to_rgb(h, s, l))

    def _on_color_picked(self, color):
        """Handle color pick — records undo."""
        if isinstance(color, list):
            color = color[0]
        old = self.current_color
        cmd = ColorChangeCommand(self, old, color, f"Pick {ColorUtils.rgb_to_hex(*color)}")
        self.undo_stack.push(cmd)

    def _set_color_internal(self, color, record=True):
        """Internal setter (used by undo/redo)."""
        self.current_color = color
        self._update_all()

    def _on_manual_color_changed(self, color):
        old = self.current_color
        cmd = ColorChangeCommand(self, old, color, "Manual input")
        self.undo_stack.push(cmd)

    def _on_color_hover(self, color):
        self.lbl_hex.setText(ColorUtils.rgb_to_hex(*color))
        self.lbl_rgb.setText(f"rgb({color[0]}, {color[1]}, {color[2]})")

    def _on_temp_changed(self, val):
        self.temp_lbl.setText(f"{val}K")
        rgb = ColorUtils.kelvin_to_rgb(val)
        self.temp_swatch.set_color(rgb)

    def _on_image_palette(self, palette):
        # Add extracted colors to saved palette
        for c in palette:
            if len(self.saved_colors) < self.max_saved:
                self.saved_colors.append(c)
        self._update_all()

    def _update_all(self):
        c = self.current_color
        self.big_swatch.set_color(c)
        self.color_input.set_color(c)
        self.lbl_hex.setText(ColorUtils.rgb_to_hex(*c))
        self.lbl_rgb.setText(f"rgb({c[0]}, {c[1]}, {c[2]})")
        h, s, l = ColorUtils.rgb_to_hsl(*c)
        self.lbl_hsl.setText(f"hsl({int(h*360)}°, {int(s*100)}%, {int(l*100)}%)")
        hv, sv, vv = ColorUtils.rgb_to_hsv(*c)
        self.lbl_hsv.setText(f"hsv({int(hv*360)}°, {int(sv*100)}%, {int(vv*100)}%)")

        # WCAG contrast
        cw = ColorUtils.contrast_ratio(c, (255, 255, 255))
        cb = ColorUtils.contrast_ratio(c, (0, 0, 0))
        self.lbl_contrast_white.setText(f"vs White: {cw:.2f}:1 — {ColorUtils.wcag_rating(cw)}")
        self.lbl_contrast_black.setText(f"vs Black: {cb:.2f}:1 — {ColorUtils.wcag_rating(cb)}")

        # Harmony
        analogous_angle = self.harmony_sliders.get("analogous")
        analogous_angle = analogous_angle.value() if analogous_angle else 30
        harmonies = {
            "complementary": [c, ColorUtils.complementary(c)],
            "analogous": ColorUtils.analogous(c, analogous_angle),
            "triadic": ColorUtils.triadic(c),
            "tetradic": ColorUtils.tetradic(c),
            "split_complementary": ColorUtils.split_complementary(c),
            "monochromatic": ColorUtils.monochromatic(c, 5),
        }
        for key, colors in harmonies.items():
            swatches = self.harmony_rows[key]
            for i, sw in enumerate(swatches):
                if i < len(colors):
                    sw.set_color(colors[i])
                    sw.setEnabled(True)
                else:
                    sw.setEnabled(False)

        # Shades
        self._update_shades()

        # CB preview
        for mode, (sw, hex_lbl) in self.cb_preview_rows.items():
            sim = ColorUtils.simulate_colorblindness(c, mode)
            sw.set_color(sim)
            hex_lbl.setText(ColorUtils.rgb_to_hex(*sim))

        # WCAG contrast previews
        for preview, info, bg in self.contrast_previews:
            ratio = ColorUtils.contrast_ratio(c, bg)
            rating = ColorUtils.wcag_rating(ratio)
            preview.setStyleSheet(
                f"background-color: rgb({bg[0]},{bg[1]},{bg[2]}); "
                f"color: rgb({c[0]},{c[1]},{c[2]}); "
                f"font-size: 15px; font-weight: bold; border-radius: 4px;")
            color_tag = "#4c4" if rating in ("AA", "AAA") else ("#cc4" if rating == "AA Large" else "#c44")
            info.setText(f"{ratio:.2f}:1 — <span style='color:{color_tag};font-weight:bold;'>{rating}</span>")

        # APCA previews
        for preview, info, bg in self.apca_previews:
            lc = ColorUtils.apca_contrast(c, bg)
            rating = ColorUtils.apca_rating(lc)
            preview.setStyleSheet(
                f"background-color: rgb({bg[0]},{bg[1]},{bg[2]}); "
                f"color: rgb({c[0]},{c[1]},{c[2]}); "
                f"font-size: 15px; font-weight: bold; border-radius: 4px;")
            color_tag = "#4c4" if rating in ("Platinum", "Silver", "Pass") else ("#cc4" if rating == "Marginal" else "#c44")
            info.setText(f"Lc {lc:.1f} — <span style='color:{color_tag};font-weight:bold;'>{rating}</span>")

        # Mixer
        self._update_mixer()

        # Palette
        for i, sw in enumerate(self.palette_swatches):
            if i < len(self.saved_colors):
                sw.set_color(self.saved_colors[i])
                sw.setEnabled(True)
            else:
                sw.set_color((40, 40, 60))
                sw.setEnabled(False)

        # History
        self._refresh_history()

    def _update_shades(self):
        steps = self.spin_steps.value()
        c = self.current_color
        for attr, func in [("shades", ColorUtils.shades),
                            ("tints", ColorUtils.tints),
                            ("tones", ColorUtils.tones)]:
            colors = func(c, steps)
            swatches = getattr(self, f"shades_{attr}")
            for i, sw in enumerate(swatches):
                if i < steps:
                    sw.set_color(colors[i])
                    sw.setEnabled(True)
                    sw.show()
                else:
                    sw.hide()

    def _update_mixer(self):
        ratio = self.mix_slider.value() / 100.0
        self.mix_ratio_lbl.setText(f"{int(ratio*100)}%")
        self.mix_swatch1.set_color(self.current_color)
        self.mix_swatch2.set_color(self.secondary_color)
        mixed = ColorUtils.mix_colors(self.current_color, self.secondary_color, ratio)
        self.mix_result.set_color(mixed)

    def _save_current(self):
        if len(self.saved_colors) >= self.max_saved:
            self.saved_colors.pop(0)
        self.saved_colors.append(self.current_color)
        self._update_all()
        self.status.showMessage("💾 Color saved", 2000)

    def _remove_from_palette(self, idx):
        if idx < len(self.saved_colors):
            self.saved_colors.pop(idx)
            self._update_all()

    def _clear_palette(self):
        self.saved_colors.clear()
        self._update_all()

    def _clear_history(self):
        self.history.clear()
        self._refresh_history()

    def _refresh_history(self):
        while self.history_layout.count() > 1:
            item = self.history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for color in self.history:
            row = QHBoxLayout()
            sw = ColorSwatch(color, size=24)
            sw.clicked.connect(self._on_color_picked)
            hex_lbl = QLabel(ColorUtils.rgb_to_hex(*color))
            hex_lbl.setStyleSheet("font-family: monospace; color: #888;")
            row.addWidget(sw); row.addWidget(hex_lbl); row.addStretch()
            container = QWidget()
            container.setLayout(row)
            self.history_layout.insertWidget(0, container)

    def _copy(self, text):
        QGuiApplication.clipboard().setText(text)
        self.status.showMessage(f"📋 Copied: {text}", 2000)

    # ------------------------------------------------------------------
    # Palette helpers
    # ------------------------------------------------------------------
    def _get_all_palette_colors(self):
        colors = [self.current_color] + self.saved_colors[:]
        for key, swatches in self.harmony_rows.items():
            for sw in swatches:
                if sw.isEnabled():
                    colors.append(sw.color)
        seen = set()
        unique = []
        for c in colors:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    # ------------------------------------------------------------------
    # Export functions
    # ------------------------------------------------------------------
    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "palette.png", "PNG (*.png)")
        if not path: return
        colors = self._get_all_palette_colors()
        size = 120
        cols = min(6, len(colors))
        rows = (len(colors) + cols - 1) // cols
        img = QImage(cols * size, rows * size, QImage.Format_RGB32)
        painter = QPainter(img)
        for i, c in enumerate(colors):
            painter.fillRect((i % cols) * size, (i // cols) * size, size, size, QColor(*c))
            painter.setPen(QColor(255, 255, 255) if sum(c) < 384 else QColor(0, 0, 0))
            painter.setFont(QFont("Consolas", 10))
            painter.drawText((i % cols) * size + 8, (i // cols) * size + size - 10,
                              ColorUtils.rgb_to_hex(*c))
        painter.end()
        img.save(path)
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    def _export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "palette.json", "JSON (*.json)")
        if not path: return
        colors = self._get_all_palette_colors()
        data = {
            "name": "Color Wheel Palette",
            "generated": datetime.now().isoformat(),
            "base_color": {"hex": ColorUtils.rgb_to_hex(*self.current_color),
                           "rgb": list(self.current_color)},
            "colors": [{"hex": ColorUtils.rgb_to_hex(*c), "rgb": list(c)} for c in colors]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    def _import_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import JSON", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path) as f:
                data = json.load(f)
            imported = []
            for c in data.get("colors", []):
                if "rgb" in c:
                    imported.append(tuple(c["rgb"]))
                elif "hex" in c:
                    rgb = ColorUtils.hex_to_rgb(c["hex"])
                    if rgb: imported.append(rgb)
            self.saved_colors = imported[:self.max_saved]
            self._update_all()
            self.status.showMessage(f"✅ Imported {len(imported)} colors", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))

    def _export_css(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSS", "palette.css", "CSS (*.css)")
        if not path: return
        colors = self._get_all_palette_colors()
        lines = ["/* Generated by 3D Color Wheel Studio */",
                 f"/* {datetime.now().strftime('%Y-%m-%d %H:%M')} */", "", ":root {"]
        lines.append(f"  --color-base: {ColorUtils.rgb_to_hex(*self.current_color)};")
        for i, c in enumerate(colors):
            lines.append(f"  --color-{i+1}: {ColorUtils.rgb_to_hex(*c)};")
        lines.append("}")
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    def _export_scss(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export SCSS", "palette.scss", "SCSS (*.scss)")
        if not path: return
        colors = self._get_all_palette_colors()
        lines = ["// Generated by 3D Color Wheel Studio",
                 f"// {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
                 f"$color-base: {ColorUtils.rgb_to_hex(*self.current_color)};",
                 "$colors: ("]
        for i, c in enumerate(colors):
            lines.append(f"  'color-{i+1}': {ColorUtils.rgb_to_hex(*c)},")
        lines.append(");")
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    def _export_tailwind(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Tailwind", "tailwind.colors.js", "JS (*.js)")
        if not path: return
        colors = self._get_all_palette_colors()
        lines = ["// Generated by 3D Color Wheel Studio",
                 "module.exports = {", "  colors: {"]
        lines.append(f"    base: '{ColorUtils.rgb_to_hex(*self.current_color)}',")
        for i, c in enumerate(colors):
            lines.append(f"    'color-{i+1}': '{ColorUtils.rgb_to_hex(*c)}',")
        lines.append("  },")
        lines.append("};")
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    def _export_swift(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Swift", "Palette.swift", "Swift (*.swift)")
        if not path: return
        colors = self._get_all_palette_colors()
        lines = ["// Generated by 3D Color Wheel Studio",
                 "import SwiftUI", "",
                 "extension Color {"]
        lines.append(f"    static let base = Color(red: {self.current_color[0]/255:.3f}, "
                     f"green: {self.current_color[1]/255:.3f}, blue: {self.current_color[2]/255:.3f})")
        for i, c in enumerate(colors):
            lines.append(f"    static let color{i+1} = Color(red: {c[0]/255:.3f}, "
                         f"green: {c[1]/255:.3f}, blue: {c[2]/255:.3f})")
        lines.append("}")
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    def _export_kotlin(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Kotlin", "Palette.kt", "Kotlin (*.kt)")
        if not path: return
        colors = self._get_all_palette_colors()
        lines = ["// Generated by 3D Color Wheel Studio",
                 "import androidx.compose.ui.graphics.Color", "",
                 "object Palette {"]
        lines.append(f"    val Base = Color(0x{self.current_color[0]:02X}{self.current_color[1]:02X}{self.current_color[2]:02X}FF)")
        for i, c in enumerate(colors):
            lines.append(f"    val Color{i+1} = Color(0x{c[0]:02X}{c[1]:02X}{c[2]:02X}FF)")
        lines.append("}")
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    def _export_android(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Android XML", "colors.xml", "XML (*.xml)")
        if not path: return
        colors = self._get_all_palette_colors()
        lines = ["<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                 "<!-- Generated by 3D Color Wheel Studio -->",
                 "<resources>"]
        lines.append(f"    <color name=\"color_base\">#{self.current_color[0]:02X}{self.current_color[1]:02X}{self.current_color[2]:02X}</color>")
        for i, c in enumerate(colors):
            lines.append(f"    <color name=\"color_{i+1}\">#{c[0]:02X}{c[1]:02X}{c[2]:02X}</color>")
        lines.append("</resources>")
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    def _export_svg(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "palette.svg", "SVG (*.svg)")
        if not path: return
        colors = self._get_all_palette_colors()
        size = 100
        cols = min(6, len(colors))
        rows = (len(colors) + cols - 1) // cols
        w, h = cols * size, rows * size
        svg = [f'<?xml version="1.0" encoding="UTF-8"?>',
               f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
               f'<!-- Generated by 3D Color Wheel Studio -->']
        for i, c in enumerate(colors):
            x = (i % cols) * size
            y = (i // cols) * size
            svg.append(f'  <rect x="{x}" y="{y}" width="{size}" height="{size}" '
                       f'fill="{ColorUtils.rgb_to_hex(*c)}"/>')
            text_color = "#FFFFFF" if sum(c) < 384 else "#000000"
            svg.append(f'  <text x="{x+8}" y="{y+size-10}" fill="{text_color}" '
                       f'font-family="monospace" font-size="12">{ColorUtils.rgb_to_hex(*c)}</text>')
        svg.append('</svg>')
        with open(path, 'w') as f:
            f.write("\n".join(svg))
        self.status.showMessage(f"✅ Exported to {path}", 3000)

    # ------------------------------------------------------------------
    def _show_shortcuts(self):
        QMessageBox.information(self, "Keyboard Shortcuts",
            "Ctrl+Z / Ctrl+Y — Undo / Redo\n"
            "Ctrl+C — Copy HEX\n"
            "Ctrl+P — Export as PNG\n"
            "Ctrl+Q — Quit\n"
            "Ctrl+R — Reset View\n"
            "Ctrl+T — Toggle Theme\n"
            "F11 — Fullscreen\n"
            "Space — Random Color\n"
            "W — Toggle Wireframe")

    def _show_about(self):
        QMessageBox.about(self, "About",
            "<h2>3D Color Wheel Studio</h2>"
            "<p>A professional color workstation.</p>"
            "<p>Built with PySide6 + OpenGL</p>"
            "<p>Features: 3D shapes, harmony, gradients, image palette extraction, "
            "named colors, undo/redo, 9 export formats, APCA contrast</p>")


# ============================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("3D Color Wheel Studio")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()