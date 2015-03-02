//Particle System
//Authors: Neeraj Jain & Ujwal Naik
//Date : 10/09/2012
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <oclUtils.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <GL/gl.h>
#include <CL/cl_gl.h>
#include "vverlet.h"

#define NUMBER_OF_PARTICLES (WIDTH*HEIGHT)
#define DATA_SIZE (NUMBER_OF_PARTICLES*4*sizeof(float))

GLuint OGL_VBO = 1;

//OpenCL vars
cl_mem oclposition,oclvelocity,oclseed;
cl_kernel mykernel;
cl_program myprogram;
cl_command_queue mycommandqueue;
cl_context mycontext;

size_t worksize[1] = {WIDTH*HEIGHT};
size_t lws[1] = {256};
float host_position[NUMBER_OF_PARTICLES][4];
float host_velocity[NUMBER_OF_PARTICLES][4];
float seed[NUMBER_OF_PARTICLES];

float left[][3] = {{2.0,0.6,2.0},{1.0,1.1,2.0},{1.0,1.1,1.0},{2.0,0.6,1.0}};
float right[][3] = {{1.5,0.1,2.0},{2.5,0.6,2.0},{2.5,0.6,1.0},{1.5,0.1,1.0}};
float bottom[][3] = {{0.0,-0.5,0.0},{0.0,-0.5,3.0},{3.0,-0.5,3.0},{3.0,-0.5,0.0}};

void do_kernel() {
	cl_event wlist[1];
	clEnqueueNDRangeKernel(mycommandqueue,mykernel,1,NULL,worksize,lws,0,0,&wlist[0]);
	clWaitForEvents(1,wlist);
}

void load_texture(char *filename,unsigned int tid)
{
	FILE *fptr;
	char buf[512], *parse, *c;
	int im_size,im_width,im_height,max_color;
	unsigned char *texture_bytes;
	size_t fsize;

	fptr = fopen(filename,"r");
	c = fgets(buf,512,fptr);
	do{
		c = fgets(buf,512,fptr);
		} while(buf[0]=='#');
	parse = strtok(buf," \t");
	im_width=atoi(parse);

	parse = strtok(NULL," \n");
	im_height = atoi(parse);

	c = fgets(buf,512,fptr);
	parse = strtok(buf," \n");
	max_color = atoi(parse);
	im_size = im_width*im_height;
	texture_bytes = (unsigned char *)calloc(3,im_size);
	fsize = fread(texture_bytes,3,im_size,fptr);
	fclose(fptr);

	glBindTexture(GL_TEXTURE_2D,tid);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,im_width,im_height,0,GL_RGB,GL_UNSIGNED_BYTE,texture_bytes);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	cfree(texture_bytes);
}

void mydisplayfunc() {
	glFinish();
	clEnqueueAcquireGLObjects(mycommandqueue,1,&oclposition,0,0,0);
	do_kernel();
	clEnqueueReleaseGLObjects(mycommandqueue,1,&oclposition,0,0,0);
	clFinish(mycommandqueue);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	int i;
	float mytexcoords[4][2] = {{0.0,1.0},{1.0,1.0},{1.0,0.0},{0.0,0.0}};
	glBindTexture(GL_TEXTURE_2D,1);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glBegin(GL_QUADS);
	glNormal3f(0.0,1.0,0.0);
	for(i=0;i<4;i++) {
		glTexCoord2fv(mytexcoords[i]);
		glVertex3fv(bottom[i]);
	}
	glEnd();
	glDisable(GL_TEXTURE_2D);	

	glBindTexture(GL_TEXTURE_2D,2);
	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);
	glNormal3f(-0.5,1.0,0.0);
	glTranslatef(1.0,0.0,0.0);
	for(i=0;i<4;i++){
		glTexCoord2fv(mytexcoords[i]);
		glVertex3fv(right[i]);
	}

	glNormal3f(0.5,1.0,0.0);
	for(i=0;i<4;i++) {
		glTexCoord2fv(mytexcoords[i]);
		glVertex3fv(left[i]);
	}
	
	glNormal3f(0.0,1.0,0.0);
	for(i=0;i<4;i++) 
		glVertex3fv(bottom[i]);

	glEnd();
	glDisable(GL_TEXTURE_2D);
	glFlush();

	glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
	glVertexPointer(4,GL_FLOAT,0,0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS,0,NUMBER_OF_PARTICLES);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();
	glutPostRedisplay();
}

float eye[] = {1.5,1.5,4.5};
float view[] = {1.5,0.0,0.0};
float up[] = {0.0,1.0,0.0};

void setup_the_viewvol() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0,1.3,0.1,20.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye[0],eye[1],eye[2],view[0],view[1],view[2],up[0],up[1],up[2]);
}

void do_lights() {
        float light_ambient[] = { 0.0,0.0,0.0,0.0 };
        float light_diffuse[] = { 1.0,1.0,1.0,0.0 };
        float light_specular[] = { 1.0,1.0,1.0,0.0 };
        float light_position[] = { 3.0,3.0,3.0,1.0 };
        float light_direction[] = { -3.0,-3.0,-3.0,1.0 };

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT,light_ambient);

        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,1);

        glLightfv(GL_LIGHT0, GL_AMBIENT,light_ambient);
        glLightfv(GL_LIGHT0, GL_DIFFUSE,light_diffuse);
        glLightfv(GL_LIGHT0, GL_SPECULAR,light_specular);
        glLightf(GL_LIGHT0, GL_SPOT_EXPONENT,1.0);
        glLightf(GL_LIGHT0, GL_SPOT_CUTOFF,180.0);
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION,0.5);
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION,0.0);
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION,0.4);
        glLightfv(GL_LIGHT0, GL_POSITION,light_position);
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION,light_direction);

        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
}

void do_material() {
        float mat_ambient[] = {0.0,0.0,0.0,1.0};
        float mat_diffuse[] = {2.8,2.8,0.4,1.0};
        float mat_specular[] = {1.5,1.5,1.5,1.0};
        float mat_shininess[] = {2.0};

        glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
        glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
        glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
        glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);
}

void InitGL(int argc, char** argv) {
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_DOUBLE);
	glutInitWindowSize(1024,768);
	glutInitWindowPosition(100,50);
	glutCreateWindow("Particle System");
	setup_the_viewvol();
	do_lights();
	do_material();
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.1,0.2,0.35,0.0);
	glewInit();
	return;
}

double genrand() {
	return (((double)(rand())+1.0) / ((double)(RAND_MAX) +2.0));
}

void Init_pos_vel_particles() {
	int i,j;
	srandom(123456789);
	for(i=0;i<NUMBER_OF_PARTICLES;i++) {
		host_position[i][0] = genrand()+1.0;
		host_position[i][1] = 0.4*genrand() + 1.1; 
		host_position[i][2] = genrand()+1.0;
		host_position[i][3] = 0.0;
		for(j=0;j<4;j++)
			host_velocity[i][j] = 0.0;
		seed[i] = genrand();
	}
	printf("Init_pos_vel_position");
}

void InitCL() {
	cl_platform_id myplatform;
	cl_device_id *mydevice;
	
	size_t program_length;
	int err;
	unsigned int gpudevcount;
	char *oclsource;
	const char *header;
	printf("InitCL(1)\n");
	err = oclGetPlatformID(&myplatform);

	err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);

	mydevice = new cl_device_id[gpudevcount];
	err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

	cl_context_properties props[] = {
				CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
				CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
				CL_CONTEXT_PLATFORM, (cl_context_properties)myplatform,
				0};

	mycontext = clCreateContext(props, 1, &mydevice[0],NULL,NULL,&err);
	mycommandqueue = clCreateCommandQueue(mycontext,mydevice[0],0,&err);

	printf("InitCL(2)\n");
	header = oclLoadProgSource("vverlet.h","",&program_length);
	oclsource = oclLoadProgSource("vverlet1.cl",header,&program_length);
	myprogram = clCreateProgramWithSource(mycontext,1,(const char **)&oclsource,&program_length,&err);
	printf("InitCL(3)\n");
	if(err==CL_SUCCESS)
                fprintf(stderr,"build ok\n");
        else
                fprintf(stderr,"build err %d\n",err);
	clBuildProgram(myprogram,0,NULL,NULL,NULL,NULL);	
	
	mykernel = clCreateKernel(myprogram, "VVerlet", &err);
	if(err==CL_SUCCESS)
                fprintf(stderr,"build ok\n");
        else
                fprintf(stderr,"build err %d\n",err);

	printf("InitCL(4)\n");
	glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
	glBufferData(GL_ARRAY_BUFFER,DATA_SIZE,&host_position[0][0],GL_DYNAMIC_DRAW);
	oclposition = clCreateFromGLBuffer(mycontext,CL_MEM_WRITE_ONLY,OGL_VBO,&err);
	oclvelocity= clCreateBuffer(mycontext,CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,DATA_SIZE,&host_velocity[0][0],&err);
	oclseed= clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,NUMBER_OF_PARTICLES*sizeof(float),&seed[0],&err);
	clSetKernelArg(mykernel,0,sizeof(cl_mem),(void *) &oclposition);
	clSetKernelArg(mykernel,1,sizeof(cl_mem),(void *) &oclvelocity);
	clSetKernelArg(mykernel,2,sizeof(cl_mem),(void *) &oclseed);
}

void cleanup(){
	clReleaseKernel(mykernel);
	clReleaseProgram(myprogram);
	clReleaseCommandQueue(mycommandqueue);
	glBindBuffer(GL_ARRAY_BUFFER, OGL_VBO);
	glDeleteBuffers(1, &OGL_VBO);
	clReleaseMemObject(oclposition);
	clReleaseContext(mycontext);
	exit(0);
}

void getout(unsigned char key, int x, int y){
	switch(key) {
		case 'q':
			cleanup();
		default:
			break;
	}
}

int main(int argc, char **argv) {
	Init_pos_vel_particles();
	InitGL(argc, argv);
	InitCL();
	glutDisplayFunc(mydisplayfunc);
	glutKeyboardFunc(getout);
	load_texture("wood.ppm",1);
	load_texture("attr.ppm",2);
	glutMainLoop();
}
