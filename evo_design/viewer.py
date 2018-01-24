# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import sys, pygame, math, pickle
from pygame.locals import *
import numpy as np
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import shutil
# from primitive import *

from OpenGL.arrays import vbo
from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays, \
                                                  glBindVertexArray
import string
import random
def rand_string(n):
    opts = string.ascii_uppercase + string.digits
    return ''.join(random.choice(opts) for _ in range(n))

class Viewer(object):
    def __init__(self, view_size=(800, 600), background=(0.7, 0.7, 0.7, 0.0)):
        # self.bounds = bounds
        self.on = True
        self.animation = None
        self.animation_playing = False
        self.draw_grid = True

        pygame.init()
        glutInit()
        self.width = view_size[0]
        self.height = view_size[1]
        viewport = view_size

        self.surface = pygame.display.set_mode(view_size, OPENGL | DOUBLEBUF )
        # glEnable(GL_DEPTH_CLAMP)

        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)


        ambient = .2
        diffuse = .5
        # glLightfv(GL_LIGHT0, GL_POSITION, [0,0, 100, 0.0])
        glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (ambient, ambient, ambient, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (diffuse, diffuse, diffuse, 1.0))
        # glLightfv(GL_LIGHT0, GL_SPECULAR, (foo, foo, foo, 1.0))

        # glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [.6, .6, .6, 1])
        # glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1))

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        # glEnable(GL_MULTISAMPLE)

        # glDepthFunc(GL_LESS)

        glClearColor(*background)

        glShadeModel(GL_SMOOTH)
        glCullFace(GL_BACK)
        # glDisable( GL_CULL_FACE )
        # glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE )

        self.clock = pygame.time.Clock()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(90.0, self.width/float(self.height), 1, 100.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        # Transparancy?
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        glTranslated(-15, -15, -15)

        self.rx, self.ry = (0,0)
        self.tx, self.ty = (0,0)
        self.zpos = 10

        self.gl_lists = []

        make_plane(5, arrows=True)
        # make_sphere(4)
        make_cube()
        self.translation_matrix = np.identity(4)
        self.scaling_matrix = np.identity(4)

    def startDraw(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)

    def endDraw(self):
        glEndList()
        self.gl_lists.append(self.gl_list)
        self.gl_list = None

    def draw_mesh(self, mesh, offset_x=0, offset_y=0, offset_z=0):
        """this time, the vertices must be specified globally, and referenced by
        indices in order to use the list paradigm. vertices is a list of 3-vectors, and
        facets are dicts containing 'normal' (a 3-vector) 'vertices' (indices to the
        other argument) and 'color' (the color the triangle should be)."""

        # first flatten out the arrays:
        mesh['vertices'] += [ offset_x, offset_y, offset_z ]
        vertices = mesh['vertices'].flatten()
        normals  = mesh['vertice_normals'].flatten()
        findices = mesh['faces'].astype('uint32').flatten()
        eindices = mesh['edges'].astype('uint32').flatten()

        fcolors = mesh['vert_colors'].flatten()
        # ecolors = np.zeros_like(mesh['vert_colors']).flatten() + .5
        ecolors = fcolors - .1
        ecolors[ecolors <= 0] = 0

        # then convert to OpenGL / ctypes arrays:
        fvertices = (GLfloat * len(vertices))(*vertices)
        evertices = (GLfloat * len(vertices))(*(vertices + normals*.001))
        normals = (GLfloat * len(normals))(*normals)
        findices = (GLuint * len(findices))(*findices)
        eindices = (GLuint * len(eindices))(*eindices)
        fcolors = (GLfloat * len(fcolors))(*fcolors)
        ecolors = (GLfloat * len(ecolors))(*ecolors)

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, fvertices)
        glNormalPointer(GL_FLOAT, 0, normals)
        glColorPointer(3, GL_FLOAT, 0, fcolors)
        glDrawElements(GL_TRIANGLES, len(findices), GL_UNSIGNED_INT, findices)

        glColorPointer(3, GL_FLOAT, 0, ecolors)
        glVertexPointer(3, GL_FLOAT, 0, evertices)
        glDrawElements(GL_LINES, len(eindices), GL_UNSIGNED_INT, eindices)
        glPopClientAttrib()

    # def draw_lines(self, lines, width=3, color=(0, 0, 0)):
    #     glLineWidth(width)

    #     glColor3f(*color)
    #     glBegin(GL_LINES)

    #     for p1, p2 in lines:
    #         glVertex3f(*p1)
    #         glVertex3f(*p2)
    #     glEnd()
    def draw_lines(self, V, E, width=3, color=(0, 0, 0)):
        glLineWidth(width)
        glColor3f(*color)
        glBegin(GL_LINES)

        for i, j in E:
            glVertex3f(*V[i])
            glVertex3f(*V[j])
        glEnd()

    def drawCube(self, p, s=1, color=(.5, .5, .5, .5)):
        glPushMatrix()
        self.translation_matrix[0, 3] = p[0]
        self.translation_matrix[1, 3] = p[1]
        self.translation_matrix[2, 3] = p[2]
        self.translation_matrix[0, 0] = s
        self.translation_matrix[1, 1] = s
        self.translation_matrix[2, 2] = s
        self.translation_matrix[3, 3] = 1
        glMultMatrixf(np.transpose(self.translation_matrix))
        glColor4f(*color)
        glCallList(G_OBJ_CUBE)
        glPopMatrix()

    def draw_text(self, x, y, text, r=0.0, g=0.0, b=0.0):
        y = self.height - (y + 24)
        glWindowPos2f(x, y)
        glColor3f(r, g, b)

        for c in text:
            if c=='\n':
                glRasterPos2f(x, y-0.24)
            else:
                glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(c))

    def drawSphere(self, p, r, color=(.5, .5, .5)):
        glPushMatrix()

        self.translation_matrix[0, 3] = p[0]
        self.translation_matrix[1, 3] = p[1]
        self.translation_matrix[2, 3] = p[2]

        self.translation_matrix[0, 0] = r
        self.translation_matrix[1, 1] = r
        self.translation_matrix[2, 2] = r
        self.translation_matrix[3, 3] = 1

        glMultMatrixf(np.transpose(self.translation_matrix))
        glColor3f(color[0], color[1], color[2])

        emmision = False
        glCallList(G_OBJ_SPHERE)
        glPopMatrix()


    # def draw_mesh(self, V, E, offset_x=0, offset_y=0, offset_z=0):
    #     # first flatten out the arrays:
    #     V += [ offset_x, offset_y, offset_z ]
    #     vertices = mesh['vertices'].flatten()
    #     # normals  = mesh['vertice_normals'].flatten()
    #     findices = mesh['faces'].astype('uint32').flatten()
    #     eindices = mesh['edges'].astype('uint32').flatten()

    #     # fcolors = mesh['vert_colors'].flatten()
    #     # ecolors = np.zeros_like(mesh['vert_colors']).flatten() + .5
    #     # ecolors = fcolors - .1
    #     # ecolors[ecolors <= 0] = 0

    #     # then convert to OpenGL / ctypes arrays:
    #     fvertices = (GLfloat * len(vertices))(*vertices)
    #     evertices = (GLfloat * len(vertices))(*(vertices))
    #     # normals = (GLfloat * len(normals))(*normals)
    #     # findices = (GLuint * len(findices))(*findices)
    #     eindices = (GLuint * len(eindices))(*eindices)
    #     # fcolors = (GLfloat * len(fcolors))(*fcolors)
    #     ecolors = (GLfloat * len(ecolors))(*ecolors)

    #     glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
    #     glEnableClientState(GL_VERTEX_ARRAY)
    #     # glEnableClientState(GL_NORMAL_ARRAY)
    #     # glEnableClientState(GL_COLOR_ARRAY)
    #     glVertexPointer(3, GL_FLOAT, 0, fvertices)
    #     # glNormalPointer(GL_FLOAT, 0, normals)
    #     # glColorPointer(3, GL_FLOAT, 0, fcolors)
    #     # glDrawElements(GL_TRIANGLES, len(findices), GL_UNSIGNED_INT, findices)

    #     # glColorPointer(3, GL_FLOAT, 0, ecolors)
    #     glVertexPointer(3, GL_FLOAT, 0, evertices)
    #     glDrawElements(GL_LINES, len(eindices), GL_UNSIGNED_INT, eindices)
    #     glPopClientAttrib()

    def clear(self):
        self.gl_lists = []

    def handle_input(self, e):
        if e.type == QUIT:
            self.on = False

        elif e.type == KEYDOWN and e.key == K_ESCAPE:
            self.on = False

        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4: self.zpos = max(1, self.zpos-1)
            elif e.button == 5: self.zpos += 1
            elif e.button == 1: self.rotate = True
            elif e.button == 3: self.move = True

        elif e.type == MOUSEBUTTONUP:
            if e.button == 1: self.rotate = False
            elif e.button == 3: self.move = False

        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if self.rotate:
                self.rx += i
                self.ry += j
            if self.move:
                self.tx += i
                self.ty -= j

        if e.type == KEYDOWN:
            if e.key == K_g:
                self.draw_grid = not self.draw_grid

    def step(self, i):
        pass

    def draw_step(self):
        pass

    def mainLoop(self, draw_func=None):
        self.rotate = False
        self.move = False
        i = 0

        while self.on:
            self.clock.tick(30)
            self.step(i)

            for e in pygame.event.get():
                self.handle_input(e)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # RENDER OBJECT
            glTranslate(self.tx/20., self.ty/20., - self.zpos)
            glRotate(self.ry, 1, 0, 0)
            glRotate(self.rx, 0, 1, 0)

            for gl_list in self.gl_lists:

                glCallList(gl_list)

            glLineWidth(1)
            if self.draw_grid:
                glCallList(G_OBJ_PLANE)

            if draw_func:
                draw_func()
            self.draw_step()

            pygame.display.flip()
            i += 1

    def save(self, path):
        pygame.image.save(self.surface, path)
