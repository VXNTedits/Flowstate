
import glm
from OpenGL.GL import *

class Renderer:
    def __init__(self, shader, camera):
        self.shader = shader
        self.camera = camera
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, 800, 600)  # Set the viewport
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set clear color (black)

    def render(self, model, view_matrix, projection_matrix):
        self.shader.use()
        model_matrix = model.model_matrix
        self.update_uniforms(model_matrix, view_matrix, projection_matrix, model)
        model.draw()

    def update_uniforms(self, model_matrix, view_matrix, projection_matrix, model):
        self.shader.set_uniform_matrix4fv("model", model_matrix)
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)
        self.shader.set_uniform3f("lightPos", glm.vec3(1.2, 1.0, 2.0))
        self.shader.set_uniform3f("viewPos", self.camera.position)
        self.shader.set_uniform3f("lightColor", glm.vec3(1, 1, 1))  # Neon purple light

        # Set material properties using the first material found
        kd = model.default_material['diffuse']
        ks = model.default_material['specular']
        ns = model.default_material['shininess']

        self.shader.set_uniform3f("objectColor", glm.vec3(*kd))
        self.shader.set_uniform3f("specularColor", glm.vec3(*ks))
        self.shader.set_uniform1f("shininess", ns)
    def draw_model_bounding_box(self, model, view_matrix, projection_matrix):
        # Use a simple shader program for drawing the bounding box
        glUseProgram(self.shader.id)
        self.shader.set_uniform_matrix4fv("model", model.model_matrix)
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)

        # Save current polygon mode and set line width
        glPushAttrib(GL_POLYGON_BIT)
        glLineWidth(3.0)  # Set the line width for the bounding box

        # Draw the wireframe over the model
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor3f(1.0, 1.0, 1.0)  # Set the color for the bounding box (white)
        model.draw()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopAttrib()  # Restore previous polygon mode