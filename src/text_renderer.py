from PIL import Image, ImageDraw, ImageFont
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os

from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *
import numpy as np


from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *
import numpy as np


class TextRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = ImageFont.truetype("arial.ttf", 24)
        self.texture = glGenTextures(1)

    def render_text(self, text, x, y):
        image = Image.new('RGB', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text((x, y), text, font=self.font, fill=(255, 255, 255, 255))

        text_data = np.array(image)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, text_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glEnable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x, y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x + self.width, y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x + self.width, y + self.height)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x, y + self.height)
        glEnd()

        glDisable(GL_TEXTURE_2D)

