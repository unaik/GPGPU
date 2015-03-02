#define omega(x,y) omega[x+y*DIRECTIONS]
#define from(i,j,k) from[(i)*(WIDTH*DIRECTIONS)+(k)*WIDTH+(j)]
#define to(i,j,k) to[(i)*(WIDTH*DIRECTIONS)+(k)*WIDTH+(j)] 


__kernel void update(__global float8* from, __global float8* to, __global int* dist, __constant float8* omega)
{
int j = get_global_id(0);
int i = get_global_id(1);
int y,x,outptr ;

float8 a[DIRECTIONS];

for(x=0;x<DIRECTIONS;x++)
{a[x]=from[(i)*(WIDTH*DIRECTIONS)+(x)*WIDTH+(j)];}

float8 new_density;
//unrolled nested for loops	

          new_density = (float8)(0.0);
	  new_density += omega(0,0)*a[0];
	  new_density += omega(1,0)*a[1];
	  new_density += omega(2,0)*a[2];
	  new_density += omega(3,0)*a[3];
	  new_density += omega(4,0)*a[4];
	  outptr=dist(i,j,0);
          to[outptr] += a[0] + new_density;

          new_density = (float8)(0.0);
	  new_density += omega(0,1)*a[0];
	  new_density += omega(1,1)*a[1];
	  new_density += omega(2,1)*a[2];
	  new_density += omega(3,1)*a[3];
	  new_density += omega(4,1)*a[4];
	  outptr=dist(i,j,1);
          to[outptr] += a[1] + new_density;

          new_density = (float8)(0.0);
	  new_density += omega(0,2)*a[0];
	  new_density += omega(1,2)*a[1];
	  new_density += omega(2,2)*a[2];
	  new_density += omega(3,2)*a[3];
	  new_density += omega(4,2)*a[4];
	  outptr=dist(i,j,2);
          to[outptr] += a[2] + new_density;

          new_density = (float8)(0.0);
	  new_density += omega(0,3)*a[0];
	  new_density += omega(1,3)*a[1];
	  new_density += omega(2,3)*a[2];
	  new_density += omega(3,3)*a[3];
	  new_density += omega(4,3)*a[4];
	  outptr=dist(i,j,3);
          to[outptr] += a[3] + new_density;

          new_density = (float8)(0.0);
	  new_density += omega(0,4)*a[0];
	  new_density += omega(1,4)*a[1];
	  new_density += omega(2,4)*a[2];
	  new_density += omega(3,4)*a[3];
	  new_density += omega(4,4)*a[4];
	  outptr=dist(i,j,4);
          to[outptr] += a[4] + new_density;



  for(y=0;y<DIRECTIONS;y++)
  {from(i,j,y) = (0.0);}
}


__kernel void heights(__global float* rbuff, __global float8* to)
{
  int i = get_global_id(1);
  int j= get_global_id(0);
  float height=0.0;
//   unrolled for loop for heights in 5 directions
  {
	height += (to(i,j,0).s0 + to(i,j,0).s1 +to(i,j,0).s2 +to(i,j,0).s3 +to(i,j,0).s4 +to(i,j,0).s5 +to(i,j,0).s6 +to(i,j,0).s7);
	height += (to(i,j,1).s0 + to(i,j,1).s1 +to(i,j,1).s2 +to(i,j,1).s3 +to(i,j,1).s4 +to(i,j,1).s5 +to(i,j,1).s6 +to(i,j,1).s7);
	height += (to(i,j,2).s0 + to(i,j,2).s1 +to(i,j,2).s2 +to(i,j,2).s3 +to(i,j,2).s4 +to(i,j,2).s5 +to(i,j,2).s6 +to(i,j,2).s7);
	height += (to(i,j,3).s0 + to(i,j,3).s1 +to(i,j,3).s2 +to(i,j,3).s3 +to(i,j,3).s4 +to(i,j,3).s5 +to(i,j,3).s6 +to(i,j,3).s7);
	height += (to(i,j,4).s0 + to(i,j,4).s1 +to(i,j,4).s2 +to(i,j,4).s3 +to(i,j,4).s4 +to(i,j,4).s5 +to(i,j,4).s6 +to(i,j,4).s7);
  }
 

 {rbuff[2*3*(j+i*WIDTH) + 1] = height;  }

  if(i>0)
  {rbuff[2*3*(j+(i-1)*WIDTH) + 4] = height;}
}




__kernel void normals(__global float* rbuff, __global float4* nbuff)
{

  int i = get_global_id(1);
  int j = get_global_id(0);
  int y1,y2,y3,y4;
  float4 norm=(0.0);
  if(i>0)
  {
   y1 = 6*((j+1) + i*WIDTH) + 1;
   y2 = 6*((j-1) + i*WIDTH) + 1;
   y3 = 6*((i+1)*WIDTH + j ) + 1;
   y4 = 6*((i-1)*WIDTH + j ) + 1;
   norm.x = rbuff[y2] - rbuff[y1];
   norm.y = 2*SCALE/WIDTH;
   norm.z = (rbuff[y4] - rbuff[y3]);
   norm.w = 0.0;
   float len = sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z + norm.w*norm.w);
   norm = norm/len;  
  }  
  nbuff[i*WIDTH+j] = norm;
}



__kernel void colors(__global float* rbuff, float4 lightdirn, __global float4* eyedirn,__global float4* nbuff)
{

	
  int i = get_global_id(1);
  int j = get_global_id(0);
  float4 norm,ec;
  float4 wnec;
  ec=eyedirn[0];
  norm = nbuff[i*WIDTH+j];
  wnec.x = dot(ec,norm);
  ec=eyedirn[1];
  wnec.y = dot(ec,norm);
  ec=-eyedirn[2];
  wnec.z = dot(ec,norm);
  wnec.w = (0.0);
  ec=-eyedirn[2];
  float4 lightness = max(0.0, -wnec*ec);
  float4 darkwater =(float4)(0.254,0.807,255.0,0.0);
  float4 lightwater =(float4)(0.0,0.807,0.819,0.0);
  float NL = dot(wnec,lightdirn);
  float4 color = NL*( mix(darkwater, lightwater,lightness) + (float4)(0.1)*(float4)(0.529,0.807,0.980,0.0)*(pow(1-lightness,5)));
  if(i>0 && i<LENGTH-1 )
  {
   rbuff[6*i*WIDTH + 6*(j)+COLOR_OFF] = color.x;
   rbuff[6*i*WIDTH + 6*(j)+COLOR_OFF+1] = color.y;
   rbuff[6*i*WIDTH + 6*(j)+COLOR_OFF+2] = color.z; 	
   
   rbuff[6*i*WIDTH + 6*(j)+COLOR_OFF+3] = color.x;
   rbuff[6*i*WIDTH + 6*(j)+COLOR_OFF+4] = color.y;
   rbuff[6*i*WIDTH + 6*(j)+COLOR_OFF+5] = color.z;
   }
}
























































