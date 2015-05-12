/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.

    Note - this is intended to be an example of using 3D textures
    in CUDA, not an optimized volume renderer.

    Changes
    sgg 22/3/2010
    - updated to use texture for display instead of glDrawPixels.
    - changed to render from front-to-back rather than back-to-front.
*/

#include "defs.h"

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

// CUDA Includes
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "volume.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_volume.ppm",
    NULL
};

const char *SDKsample = "CUDA 3D Volume Render";

cudaExtent volumeSize = make_cudaExtent(256, 256, 256);
typedef unsigned short VolumeType;
CHAR_GRID vdens1;

uint width1 = 900, height1 = 900;
uint xpos1 = 100, ypos1 = 100;
int deviceRend;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float delay = 0;

float density1 = 0.10f;
float brightness1 = 1.5f;
float transferOffset1 = 0.0f;
float transferScale1 = 5.0f;
float weightSlider1 = 1.0f;

bool linearFiltering = true;
bool volSwitch = true;

GLuint pbo1 = 0;     // OpenGL pixel buffer object
GLuint tex1 = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource1; // CUDA Graphics Resource (to transfer PBO)


int *pArgc;
char **pArgv;

extern "C" void cudaInitVdens( FLOAT_GRID *,
                                 CHAR_GRID *,
                                 float );
extern "C" void initCudaDens(void *, cudaExtent, int );
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 , dim3 , uint *, uint , uint ,
							  float , float , float , float , float );
extern "C" void copyInvViewMatrix(float *, size_t );

void initPixelBuffer();


// render image using CUDA
void render1( )
{
	copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output1;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource1, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output1, &num_bytes,
						       cuda_pbo_resource1));

    // clear image
    checkCudaErrors(cudaMemset(d_output1, 0, width1*height1*4));

    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output1, width1, height1, density1, brightness1, transferOffset1, transferScale1, weightSlider1);

    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource1, 0));
}


// display results using OpenGL (called by GLUT)
void display1()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
        glLoadIdentity();
        glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
        glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
        glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

    render1();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo1);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo1);
    glBindTexture(GL_TEXTURE_2D, tex1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width1, height1, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
#endif

    glutSwapBuffers();
    glutReportErrors();
}


void idle()
{
    glutPostRedisplay();
}


void keyboard1(unsigned char key, int x, int y)
{
    switch(key) {
        case 27:
            glutLeaveMainLoop();
            break;
        case 'f':
            linearFiltering = !linearFiltering;
            break;

        case '=':
          {
            density1 += 0.001f;
            break;
          }
        case '-':
          {
            density1 -= 0.001f;
            break;
          }

        case ']':
          {
            brightness1 += 0.1f;
            break;
          }

        case '[':
          {
            brightness1 -= 0.1f;
            break;
          }

        case 'l':
          {
            transferOffset1 += 0.01f;
            break;
          }
        case 'k':
          {
            transferOffset1 -= 0.01f;
            break;
          }

        case '.':
          {
            transferScale1 += 0.01f;
            break;
          }
        case ',':
          {
            transferScale1 -= 0.01f;
            break;
          }

        default:
            break;
    }
}


int ox, oy;
int buttonState = 0;


void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState  |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;

    glutPostRedisplay();
}

void motion1(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 5) {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 4) {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1) {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x; oy = y;
}


int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void cleanup()
{
    freeCudaBuffers();

    if (pbo1) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource1);
        glDeleteBuffersARB(1, &pbo1);
        glDeleteTextures(1, &tex1);
    }
}

void initPixelBuffer1()
{
    if (pbo1) {
        // unregister this buffer object from CUDA C
        //checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource1));
        // delete old buffer
        glDeleteBuffersARB(1, &pbo1);
        glDeleteTextures(1, &tex1);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo1);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo1);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width1*height1*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource1, pbo1, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex1);
    glBindTexture(GL_TEXTURE_2D, tex1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width1, height1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void reshape1(int w, int h)
{
    width1 = w; height1 = h;
    //printf("\n width = %d | height = %d\n",width1,height1);
    initPixelBuffer1();

    // calculate new grid size
    gridSize = dim3(iDivUp(width1, blockSize.x), iDivUp(height1, blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}


void display(void) {

    display1();
}

void keyboard(unsigned char key, int x, int y) {

    keyboard1(key,x,y);
}

void reshape(int w, int h) {

    reshape1(w,h);

    glutPostRedisplay();
}

void motion(int x, int y) {

    motion1(x,y);

    glutPostRedisplay();
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
vrender( FLOAT_GRID *dens1, float datamax1,
         float idens, float ibright, float ioffset, float iscale, float iweight,
         int colorScale )
{
    cudaSetDevice(0);

    density1 = idens;
    brightness1 = ibright;
    transferOffset1 = ioffset;
    transferScale1 = iscale;
    weightSlider1 = 1.f;

    deviceRend = 0;

    volumeSize.width = dens1->count.x;
    volumeSize.height = dens1->count.y;
    volumeSize.depth = dens1->count.z;

    vdens1.count.x = dens1->count.x;
    vdens1.count.y = dens1->count.y;
    vdens1.count.z = dens1->count.z;

    cudaInitVdens(dens1, &vdens1, datamax1);
    initCudaDens( vdens1.matrix, volumeSize, colorScale );

    // calculate new grid size
    gridSize = dim3(iDivUp(width1, blockSize.x), iDivUp(height1, blockSize.y));

    int c = 1;
    char *foo = "name";
    glutInit( &c, &foo );
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width1, height1);

    glutCreateWindow("CUDA Volume Render");
        glewInit();
        glutPositionWindow(xpos1,ypos1);
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutReshapeFunc(reshape);
        glutMotionFunc(motion);
        glutMouseFunc(mouse);
        initPixelBuffer1();

    glutIdleFunc(idle);
    atexit(cleanup);
    glutMainLoop();

    return(SUCCESS);
}
